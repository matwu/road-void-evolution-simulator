"""
Train and evaluate a void volume growth-rate regression model.

This is a provisional pipeline: it trains on the void geometry trajectories
recorded in metadata.yaml (see dataset.py), not on real GPR waveform data,
since no gprMax simulation output exists in this repository yet. It exists to
validate the preprocessing / feature engineering / train-eval flow end-to-end
so real GPR-derived features can be swapped in later (via a different
FeatureExtractor) without restructuring the pipeline.

Evaluation uses GroupKFold (grouped by sequence_id, so stages from the same
sequence never split across train/test) and reports RandomForest metrics
alongside a naive persistence baseline (predict the previous stage's
realized growth rate), since the dummy data's deterministic growth curve
makes a single point estimate of R2 easy to over-interpret.

Usage:
    python src/ml/train_growth_model.py path/to/metadata.yaml
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from dataset import (  # noqa: E402
    GROUP_COLUMN,
    PERSISTENCE_BASELINE_COLUMN,
    TARGET_COLUMN,
    GeometryFeatureExtractor,
    WaveformFeatureExtractor,
    build_growth_dataset,
)


def _score(y_true, y_pred) -> dict:
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': r2_score(y_true, y_pred),
    }


def _summarize(fold_scores: list) -> dict:
    """Aggregate a list of per-fold metric dicts into {metric: (mean, std)}."""
    return {
        key: (
            float(np.mean([s[key] for s in fold_scores])),
            float(np.std([s[key] for s in fold_scores])),
        )
        for key in fold_scores[0]
    }


def cross_validate(metadata_path, feature_extractor=None, n_splits=5, random_state=0, model_params=None):
    """
    Evaluate a RandomForestRegressor against a naive persistence baseline
    using GroupKFold, so stages from the same sequence never split across
    train/test.

    Returns (rf_fold_scores, baseline_fold_scores, rf_summary, baseline_summary),
    where each *_summary maps metric name -> (mean, std) across folds.
    """
    feature_extractor = feature_extractor or GeometryFeatureExtractor()
    model_params = model_params or {}

    df = build_growth_dataset(metadata_path, feature_extractor)
    n_sequences = df[GROUP_COLUMN].nunique()
    if n_sequences < n_splits:
        raise ValueError(f'Need at least n_splits={n_splits} sequences, found {n_sequences}.')

    X = df[feature_extractor.feature_columns]
    y = df[TARGET_COLUMN]
    groups = df[GROUP_COLUMN]
    baseline_pred = df[PERSISTENCE_BASELINE_COLUMN]

    rf_fold_scores = []
    baseline_fold_scores = []

    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = RandomForestRegressor(random_state=random_state, **model_params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        rf_pred = model.predict(X.iloc[test_idx])

        rf_fold_scores.append(_score(y.iloc[test_idx], rf_pred))
        baseline_fold_scores.append(_score(y.iloc[test_idx], baseline_pred.iloc[test_idx]))

    return rf_fold_scores, baseline_fold_scores, _summarize(rf_fold_scores), _summarize(baseline_fold_scores)


def train_final_model(metadata_path, feature_extractor=None, random_state=0, model_params=None):
    """Fit a model on the full dataset, for saving/deployment (not used for the reported CV metrics)."""
    feature_extractor = feature_extractor or GeometryFeatureExtractor()
    model_params = model_params or {}
    df = build_growth_dataset(metadata_path, feature_extractor)
    model = RandomForestRegressor(random_state=random_state, **model_params)
    model.fit(df[feature_extractor.feature_columns], df[TARGET_COLUMN])
    return model


def main():
    parser = argparse.ArgumentParser(description='Train/evaluate void volume growth-rate regression model')
    parser.add_argument('metadata_path', type=Path, help='Path to metadata.yaml produced by gpr_simulator.py')
    parser.add_argument('--model-output', type=Path, default=None, help='Where to save the final model (joblib)')
    parser.add_argument('--n-splits', type=int, default=5, help='Number of GroupKFold folds')
    parser.add_argument('--random-state', type=int, default=0)
    parser.add_argument('--n-estimators', type=int, default=200)
    parser.add_argument('--max-depth', type=int, default=None)
    parser.add_argument('--min-samples-leaf', type=int, default=1)
    parser.add_argument(
        '--feature-source', choices=['geometry', 'waveform'], default='geometry',
        help='geometry: ground-truth void size/depth (default). '
             'waveform: signal features extracted from real gprMax .out B-scan files.'
    )
    parser.add_argument(
        '--waveform-dir', type=Path, default=None,
        help='Directory containing the per-trace .out files (required for --feature-source waveform; '
             'defaults to the metadata.yaml directory)'
    )
    parser.add_argument(
        '--num-traces', type=int, default=None,
        help='Number of B-scan traces per stage, i.e. the -n used when running gprMax '
             '(required for --feature-source waveform)'
    )
    args = parser.parse_args()

    model_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_leaf': args.min_samples_leaf,
    }

    if args.feature_source == 'waveform':
        if args.num_traces is None:
            parser.error('--num-traces is required for --feature-source waveform')
        waveform_dir = args.waveform_dir or args.metadata_path.parent
        feature_extractor = WaveformFeatureExtractor(waveform_dir, num_traces=args.num_traces)
    else:
        feature_extractor = GeometryFeatureExtractor()

    rf_folds, baseline_folds, rf_summary, baseline_summary = cross_validate(
        args.metadata_path, feature_extractor, args.n_splits, args.random_state, model_params
    )

    print(f'GroupKFold cross-validation ({args.n_splits} folds, grouped by sequence_id)\n')
    print('RandomForest:')
    for key, (mean, std) in rf_summary.items():
        print(f'  {key}: {mean:.4f} +/- {std:.4f}')

    print("\nPersistence baseline (predict the previous stage's realized growth rate):")
    for key, (mean, std) in baseline_summary.items():
        print(f'  {key}: {mean:.4f} +/- {std:.4f}')

    r2_gain = rf_summary['r2'][0] - baseline_summary['r2'][0]
    print(f'\nRandomForest vs. persistence baseline, mean R2 gain: {r2_gain:+.4f}')
    if r2_gain <= 0:
        print('WARNING: RandomForest does not outperform the naive persistence baseline on average.')

    if args.model_output:
        final_model = train_final_model(args.metadata_path, feature_extractor, args.random_state, model_params)
        joblib.dump(
            {
                'model': final_model,
                'feature_columns': feature_extractor.feature_columns,
                'cv_summary': {'random_forest': rf_summary, 'persistence_baseline': baseline_summary},
            },
            args.model_output,
        )
        print(f'\nFinal model (trained on the full dataset) saved to {args.model_output}')


if __name__ == '__main__':
    main()
