"""
Feature engineering for void volume growth-rate prediction.

The pipeline is split into three independent stages so that today's
ground-truth geometry features (from the simulator's own metadata.yaml) can
later be swapped for real GPR waveform-derived features without touching the
label logic or the training code:

1. load_stage_records() -- reads the raw per-sequence, per-stage records.
   Today this reads metadata.yaml; a future version could instead read
   exported CSV/HDF5 waveform data, as long as it's keyed the same way
   (sequence_id, stage).
2. build_labels() -- computes the training target (volume_growth_rate) from
   ground-truth geometry. This stays geometry-based even after switching to
   waveform-derived features, since supervised training always needs known
   ground truth to learn against.
3. FeatureExtractor -- computes the model's input features for a sequence up
   to stage t. GeometryFeatureExtractor (today's features) and any future
   WaveformFeatureExtractor implement the same interface, so
   build_growth_dataset() and the training script don't need to change when
   the feature source changes.
"""
from pathlib import Path
from typing import Dict, List, Protocol

import h5py
import numpy as np
import pandas as pd
import yaml

GROUP_COLUMN = 'sequence_id'
TARGET_COLUMN = 'volume_growth_rate'
# Naive "predict the previous stage's realized growth rate" baseline, computed
# purely from label history so it stays valid regardless of which
# FeatureExtractor is in use.
PERSISTENCE_BASELINE_COLUMN = 'persistence_baseline_growth_rate'


def load_stage_records(metadata_path) -> Dict[int, List[Dict]]:
    """Load metadata.yaml and group/sort records by sequence_id, then stage."""
    with open(metadata_path, 'r') as f:
        records = yaml.safe_load(f)

    by_sequence: Dict[int, List[Dict]] = {}
    for record in records:
        by_sequence.setdefault(record['sequence_id'], []).append(record)

    return {
        sequence_id: sorted(stage_records, key=lambda r: r['stage'])
        for sequence_id, stage_records in by_sequence.items()
    }


def _void_volume(void_params: Dict) -> float:
    return void_params['size_x'] * void_params['size_y'] * void_params['size_z']


def build_labels(stage_records: List[Dict]) -> Dict[int, float]:
    """
    Compute the ground-truth volume_growth_rate label for each stage t (for
    t < last stage) of a single, already stage-sorted sequence.

    Keyed by the record's 'stage' field (not position) so callers don't need
    to assume stages are contiguous integers starting at 0.
    """
    volumes = [_void_volume(r['void_params']) for r in stage_records]
    return {
        stage_records[t]['stage']: (volumes[t + 1] - volumes[t]) / volumes[t]
        for t in range(len(stage_records) - 1)
    }


class FeatureExtractor(Protocol):
    """Interface for computing model input features at stage t of a sequence."""

    feature_columns: List[str]

    def extract(self, stage_records: List[Dict], t: int) -> Dict[str, float]:
        ...


class GeometryFeatureExtractor:
    """
    Features derived from the simulator's own ground-truth geometry.

    This is a stand-in until real gprMax B-scan output exists. A future
    WaveformFeatureExtractor reading exported CSV/HDF5 signal features (peak
    amplitude, two-way travel time, hyperbola curvature, SNR, ...) can
    implement the same extract(stage_records, t) -> Dict interface and be
    passed to build_growth_dataset() in place of this class -- nothing else
    in the pipeline needs to change.
    """

    feature_columns = [
        'stage',
        'volume',
        'size_x',
        'size_y',
        'size_z',
        'center_z',
        'volume_velocity',
        'depth_velocity',
    ]

    def extract(self, stage_records: List[Dict], t: int) -> Dict[str, float]:
        void_params = stage_records[t]['void_params']
        volume_t = _void_volume(void_params)
        depth_t = void_params['center_z']

        if t > 0:
            prev_void_params = stage_records[t - 1]['void_params']
            volume_velocity = volume_t - _void_volume(prev_void_params)
            depth_velocity = depth_t - prev_void_params['center_z']
        else:
            volume_velocity = 0.0
            depth_velocity = 0.0

        return {
            'stage': stage_records[t]['stage'],
            'volume': volume_t,
            'size_x': void_params['size_x'],
            'size_y': void_params['size_y'],
            'size_z': void_params['size_z'],
            'center_z': depth_t,
            'volume_velocity': volume_velocity,
            'depth_velocity': depth_velocity,
        }


class WaveformFeatureExtractor:
    """
    Features derived from real gprMax B-scan waveform output, replacing
    GeometryFeatureExtractor's ground-truth geometry with signal
    characteristics a real GPR system could actually measure.

    gprMax's `-n N` batch mode (used to create a B-scan) does not write a
    single merged file: for an input file `stem.in` it writes N separate
    files named `stem1.out`, `stem2.out`, ..., `stemN.out`, each containing
    one trace (group 'rx1'). This extractor reads that per-trace file set
    directly for each stage's `input_file` (as recorded in metadata.yaml).

    Processing per stage:
    1. Stack the N traces into a (traces x time) B-scan array.
    2. Subtract the mean trace (background removal) to cancel the flat,
       common-mode layer reflections shared by every trace, leaving mostly
       the void's hyperbolic reflection, which varies with trace position.
    3. Discard samples before `direct_wave_cutoff_ns`, where the direct
       source-to-receiver coupling dominates.
    4. From the residual: peak amplitude, total energy ("reflection
       intensity"), the two-way travel time of the peak (a depth proxy), and
       the fraction of traces whose peak reaches at least half the global
       peak (a hyperbola-width proxy).
    """

    feature_columns = [
        'stage',
        'peak_amplitude',
        'reflection_energy',
        'peak_time_ns',
        'hyperbola_width_ratio',
        'peak_amplitude_velocity',
        'reflection_energy_velocity',
    ]

    def __init__(self, output_dir, num_traces: int, component: str = 'Ez', direct_wave_cutoff_ns: float = 4.0):
        self.output_dir = Path(output_dir)
        self.num_traces = num_traces
        self.component = component
        self.direct_wave_cutoff_ns = direct_wave_cutoff_ns
        self._stats_cache: Dict[str, Dict[str, float]] = {}

    def _load_bscan(self, input_filename: str):
        stem = Path(input_filename).stem
        traces = []
        dt = None
        for i in range(1, self.num_traces + 1):
            out_path = self.output_dir / f'{stem}{i}.out'
            with h5py.File(out_path, 'r') as f:
                traces.append(np.array(f['rxs']['rx1'][self.component]))
                dt = float(f.attrs['dt'])
        return np.array(traces), dt

    def _stage_stats(self, input_filename: str) -> Dict[str, float]:
        if input_filename in self._stats_cache:
            return self._stats_cache[input_filename]

        traces, dt = self._load_bscan(input_filename)
        background = traces.mean(axis=0, keepdims=True)
        residual = traces - background

        cutoff_idx = int(self.direct_wave_cutoff_ns * 1e-9 / dt)
        residual = residual[:, cutoff_idx:]

        abs_residual = np.abs(residual)
        per_trace_peak = abs_residual.max(axis=1)
        peak_amplitude = float(per_trace_peak.max())
        peak_trace = int(per_trace_peak.argmax())
        peak_time_ns = float((abs_residual[peak_trace].argmax() + cutoff_idx) * dt * 1e9)
        reflection_energy = float(np.sum(residual ** 2))
        hyperbola_width_ratio = float(np.mean(per_trace_peak >= 0.5 * peak_amplitude))

        stats = {
            'peak_amplitude': peak_amplitude,
            'reflection_energy': reflection_energy,
            'peak_time_ns': peak_time_ns,
            'hyperbola_width_ratio': hyperbola_width_ratio,
        }
        self._stats_cache[input_filename] = stats
        return stats

    def extract(self, stage_records: List[Dict], t: int) -> Dict[str, float]:
        stats_t = self._stage_stats(stage_records[t]['input_file'])

        if t > 0:
            stats_prev = self._stage_stats(stage_records[t - 1]['input_file'])
            peak_amplitude_velocity = stats_t['peak_amplitude'] - stats_prev['peak_amplitude']
            reflection_energy_velocity = stats_t['reflection_energy'] - stats_prev['reflection_energy']
        else:
            peak_amplitude_velocity = 0.0
            reflection_energy_velocity = 0.0

        return {
            'stage': stage_records[t]['stage'],
            **stats_t,
            'peak_amplitude_velocity': peak_amplitude_velocity,
            'reflection_energy_velocity': reflection_energy_velocity,
        }


def build_growth_dataset(metadata_path, feature_extractor: FeatureExtractor = None) -> pd.DataFrame:
    """
    Build a (features, label, baseline) table for predicting next-stage
    volume growth rate.

    feature_extractor defaults to GeometryFeatureExtractor; pass a different
    implementation (e.g. a future waveform-based one) to change what the
    model sees without touching label or baseline computation.
    """
    feature_extractor = feature_extractor or GeometryFeatureExtractor()
    by_sequence = load_stage_records(metadata_path)

    rows = []
    for sequence_id, stage_records in by_sequence.items():
        labels = build_labels(stage_records)
        stage_order = [r['stage'] for r in stage_records]

        for t in range(len(stage_records) - 1):
            stage = stage_order[t]
            # Persistence baseline: the most recently *realized* growth rate
            # (from stage t-1 to t), used as a naive forecast for stage t to t+1.
            baseline = labels[stage_order[t - 1]] if t > 0 else 0.0

            row = {
                GROUP_COLUMN: sequence_id,
                TARGET_COLUMN: labels[stage],
                PERSISTENCE_BASELINE_COLUMN: baseline,
            }
            row.update(feature_extractor.extract(stage_records, t))
            rows.append(row)

    return pd.DataFrame(rows)
