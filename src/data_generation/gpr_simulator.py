"""
GPR Simulation: Time-series data generation for road subsurface void evolution
"""
import os
import numpy as np
from pathlib import Path
import h5py
from typing import Tuple, List, Dict
import yaml


class VoidEvolutionSimulator:
    """Class to simulate void growth and upward movement"""

    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Road structure parameters
        road = config['road']
        self.air_thickness = road['air_thickness']
        self.surface_asphalt_thickness = road['surface_asphalt_thickness']
        self.base_asphalt_thickness = road['base_asphalt_thickness']
        self.upper_subbase_thickness = road['upper_subbase_thickness']
        self.lower_subbase_thickness = road['lower_subbase_thickness']
        self.subgrade_thickness = road['subgrade_thickness']

        # GPR parameters
        self.frequency = config['gpr']['frequency']  # Center frequency (MHz)
        self.time_window = config['gpr']['time_window']  # Time window (ns)
        self.spatial_resolution = config['gpr']['spatial_resolution']  # Spatial resolution (m)

        # B-scan parameters
        self.num_traces = config['gpr'].get('num_traces', 50)  # Number of B-scan traces
        self.scan_start_x_ratio = config['gpr'].get('scan_start_x_ratio', 0.1)  # Start position ratio (0-1)
        self.scan_end_x_ratio = config['gpr'].get('scan_end_x_ratio', 0.9)  # End position ratio (0-1)

        # Void parameter ranges (all ratios 0-1)
        void = config['void']
        self.void_initial_x_position_range = void['initial_x_position_range']
        self.void_initial_y_position_range = void['initial_y_position_range']
        self.void_initial_depth_ratio_range = void['initial_depth_ratio_range']
        self.void_initial_size_x_ratio_range = void['initial_size_x_ratio_range']
        self.void_initial_size_y_ratio_range = void['initial_size_y_ratio_range']
        self.void_initial_size_z_ratio_range = void['initial_size_z_ratio_range']
        self.void_growth_rate_range = void['growth_rate_range']
        self.void_upward_movement_ratio_range = void['upward_movement_ratio_range']

        # Materials
        self.materials = config['materials']

        # Domain
        domain = config['domain']
        self.domain_x = domain['size_x']
        self.domain_y = domain['size_y']
        # Calculate domain_z from road layer thicknesses
        self.domain_z = (self.air_thickness +
                        self.surface_asphalt_thickness +
                        self.base_asphalt_thickness +
                        self.upper_subbase_thickness +
                        self.lower_subbase_thickness +
                        self.subgrade_thickness)

    def generate_void_parameters(self, stage: int, total_stages: int, sequence_seed: int = None) -> Dict:
        """
        Generate void time-evolution parameters

        Args:
            stage: Current stage (0 is initial, total_stages-1 is final)
            total_stages: Total number of stages
            sequence_seed: Seed to fix initial values per sequence

        Returns:
            Void parameter dictionary with absolute coordinates
        """
        # Progress (0.0 ~ 1.0)
        progress = stage / (total_stages - 1) if total_stages > 1 else 0

        # Generate consistent initial parameters within a sequence
        if sequence_seed is not None and stage == 0:
            np.random.seed(sequence_seed)

        # Calculate road depth (excluding air layer)
        road_depth = self.domain_z - self.air_thickness

        # Initial position and size ratios (get range from config)
        if stage == 0:
            # For stage 0, generate new initial values as ratios
            self._initial_x_position_ratio = np.random.uniform(*self.void_initial_x_position_range)
            self._initial_y_position_ratio = np.random.uniform(*self.void_initial_y_position_range)
            self._initial_depth_ratio = np.random.uniform(*self.void_initial_depth_ratio_range)
            self._initial_size_x_ratio = np.random.uniform(*self.void_initial_size_x_ratio_range)
            self._initial_size_y_ratio = np.random.uniform(*self.void_initial_size_y_ratio_range)
            self._initial_size_z_ratio = np.random.uniform(*self.void_initial_size_z_ratio_range)
            self._max_growth_rate = np.random.uniform(*self.void_growth_rate_range)
            self._max_upward_movement_ratio = np.random.uniform(*self.void_upward_movement_ratio_range)

        # Growth rate (simulate non-linear growth)
        growth_rate = 1.0 + progress ** 1.5 * (self._max_growth_rate - 1.0)

        # Upward movement as ratio of initial depth
        upward_movement_ratio = progress * self._max_upward_movement_ratio

        # Convert ratios to absolute coordinates
        center_x = self._initial_x_position_ratio * self.domain_x
        center_y = self._initial_y_position_ratio * self.domain_y
        initial_depth = self._initial_depth_ratio * road_depth
        upward_movement = upward_movement_ratio * initial_depth
        center_z = initial_depth - upward_movement  # Rising toward surface

        size_x = self._initial_size_x_ratio * self.domain_x * growth_rate
        size_y = self._initial_size_y_ratio * self.domain_y * growth_rate
        size_z = self._initial_size_z_ratio * road_depth * growth_rate ** 0.8

        return {
            'center_x': center_x,
            'center_y': center_y,
            'center_z': center_z,
            'size_x': size_x,
            'size_y': size_y,
            'size_z': size_z,
            'stage': stage,
            'progress': progress
        }

    def create_gpr_input_file(self, void_params: Dict, filename: str) -> str:
        """
        Generate gprMax input file (.in) for B-scan

        Args:
            void_params: Void parameters
            filename: Output filename

        Returns:
            Generated file path
        """
        filepath = self.output_dir / filename

        # Domain size (from config)
        domain_x = self.domain_x
        domain_y = self.domain_y
        domain_z = self.domain_z

        # Discretization (dx)
        dx = self.spatial_resolution

        # Calculate layer positions
        # z=0 is road surface (antenna position), positive z goes downward into road
        z_road_surface = 0.0  # Road surface level (antenna position)

        # Air layer extends downward (negative direction from road surface)
        z_air_bottom = z_road_surface - self.air_thickness

        # Road layers going downward from surface (positive z direction)
        z_surface_asphalt_bottom = z_road_surface + self.surface_asphalt_thickness
        z_base_asphalt_bottom = z_surface_asphalt_bottom + self.base_asphalt_thickness
        z_upper_subbase_bottom = z_base_asphalt_bottom + self.upper_subbase_thickness
        z_lower_subbase_bottom = z_upper_subbase_bottom + self.lower_subbase_thickness
        z_subgrade_bottom = z_lower_subbase_bottom + self.subgrade_thickness  # Domain bottom

        # Calculate B-scan parameters
        # Convert ratio to absolute position
        tx_start_x = self.scan_start_x_ratio * domain_x
        tx_end_x = self.scan_end_x_ratio * domain_x
        tx_y = domain_y / 2
        tx_z = 0.0  # At road surface level

        # Calculate step size for B-scan
        scan_length = tx_end_x - tx_start_x
        step_size = scan_length / (self.num_traces - 1) if self.num_traces > 1 else 0

        # Domain coordinates (gprMax requires positive values starting from 0)
        # We need to shift everything so the air layer bottom is at z=0
        z_offset = abs(z_air_bottom)  # Offset to make air bottom at z=0

        # Apply offset to all z coordinates
        z_domain_bottom = z_offset + z_air_bottom  # Should be 0.0
        z_air_top = z_offset + z_road_surface  # Road surface after offset
        z_surface_asphalt_top = z_air_top
        z_base_asphalt_top = z_offset + z_surface_asphalt_bottom
        z_upper_subbase_top = z_offset + z_base_asphalt_bottom
        z_lower_subbase_top = z_offset + z_upper_subbase_bottom
        z_subgrade_top = z_offset + z_lower_subbase_bottom
        z_domain_top = z_offset + z_subgrade_bottom  # Domain top

        # Antenna at road surface (z=0 in logical coordinates, z_air_top after offset)
        antenna_z = z_air_top

        content = f"""#title: Road void evolution stage {void_params['stage']} (B-scan)
#domain: {domain_x} {domain_y} {z_domain_top}
#dx_dy_dz: {dx} {dx} {dx}
#time_window: {self.time_window}e-9

#material: {self.materials['air']} 0 1 0 air
#material: {self.materials['surface_asphalt']} 0.01 1 0 surface_asphalt
#material: {self.materials['base_asphalt']} 0.01 1 0 base_asphalt
#material: {self.materials['upper_subbase']} 0.02 1 0 upper_subbase
#material: {self.materials['lower_subbase']} 0.02 1 0 lower_subbase
#material: {self.materials['subgrade']} 0.05 1 0 subgrade
#material: {self.materials['void']} 0 1 0 void

#box: 0 0 {z_domain_bottom} {domain_x} {domain_y} {z_air_top} air
#box: 0 0 {z_surface_asphalt_top} {domain_x} {domain_y} {z_base_asphalt_top} surface_asphalt
#box: 0 0 {z_base_asphalt_top} {domain_x} {domain_y} {z_upper_subbase_top} base_asphalt
#box: 0 0 {z_upper_subbase_top} {domain_x} {domain_y} {z_lower_subbase_top} upper_subbase
#box: 0 0 {z_lower_subbase_top} {domain_x} {domain_y} {z_subgrade_top} lower_subbase
#box: 0 0 {z_subgrade_top} {domain_x} {domain_y} {z_domain_top} subgrade

#box: {void_params['center_x'] - void_params['size_x']/2} {void_params['center_y'] - void_params['size_y']/2} {z_offset + void_params['center_z']} {void_params['center_x'] + void_params['size_x']/2} {void_params['center_y'] + void_params['size_y']/2} {z_offset + void_params['center_z'] + void_params['size_z']} void

#waveform: ricker 1 {self.frequency}e6 my_ricker
#hertzian_dipole: z {tx_start_x} {tx_y} {antenna_z} my_ricker
#rx: {tx_start_x} {tx_y} {antenna_z}

#src_steps: {step_size} 0 0
#rx_steps: {step_size} 0 0

#geometry_view: 0 0 0 {domain_x} {domain_y} {domain_z} {dx} {dx} {dx} geometry_stage_{void_params['stage']} f
"""

        with open(filepath, 'w') as f:
            f.write(content)

        return str(filepath)

    def generate_time_series_dataset(
        self,
        num_sequences: int,
        stages_per_sequence: int
    ) -> None:
        """
        Generate time-series dataset

        Args:
            num_sequences: Number of sequences to generate
            stages_per_sequence: Number of stages per sequence
        """
        print(f"Generating {num_sequences} time series sequences...")
        print(f"Each sequence has {stages_per_sequence} stages")

        metadata = []

        for seq_id in range(num_sequences):
            print(f"\nSequence {seq_id + 1}/{num_sequences}")
            sequence_metadata = []

            for stage in range(stages_per_sequence):
                # Use consistent initial parameters per sequence
                void_params = self.generate_void_parameters(stage, stages_per_sequence, sequence_seed=seq_id)

                # Generate gprMax input file
                input_filename = f"seq_{seq_id:04d}_stage_{stage:02d}.in"
                input_file = self.create_gpr_input_file(void_params, input_filename)

                sequence_metadata.append({
                    'sequence_id': seq_id,
                    'stage': stage,
                    'input_file': input_filename,
                    'void_params': void_params
                })

                print(f"  Stage {stage}: depth={void_params['center_z']:.2f}m, "
                      f"size_x={void_params['size_x']:.2f}m")

            metadata.extend(sequence_metadata)

        # Save metadata
        metadata_file = self.output_dir / 'metadata.yaml'
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)

        print(f"\nMetadata saved to {metadata_file}")
        print(f"Total input files generated: {len(metadata)}")

        return metadata


class GPRDataProcessor:
    """gprMax output data processing class"""

    @staticmethod
    def load_gpr_output(output_file: str) -> np.ndarray:
        """
        Load data from gprMax output file (.out)

        Args:
            output_file: Output file path

        Returns:
            GPR data array
        """
        with h5py.File(output_file, 'r') as f:
            # Adjust according to gprMax output structure
            rxs = f['rxs']
            rx_component = rxs['rx1']['Ez']  # E field z component
            data = np.array(rx_component)

        return data

    @staticmethod
    def normalize_data(data: np.ndarray) -> np.ndarray:
        """
        Normalize data

        Args:
            data: Input data

        Returns:
            Normalized data
        """
        mean = np.mean(data)
        std = np.std(data)

        if std > 0:
            return (data - mean) / std
        else:
            return data - mean


def main():
    """Main execution function"""

    # Load configuration file
    config_path = Path(__file__).parent.parent.parent / 'config' / 'simulation_config.yaml'

    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loaded configuration from {config_path}")
    print(f"Output directory: {config['output_dir']}")
    print(f"GPR frequency: {config['gpr']['frequency']} MHz")
    print(f"Spatial resolution: {config['gpr']['spatial_resolution']} m")
    print(f"Generating {config['generation']['num_sequences']} sequences with {config['generation']['stages_per_sequence']} stages each")

    # Initialize simulator
    simulator = VoidEvolutionSimulator(config)

    # Generate dataset (from config)
    metadata = simulator.generate_time_series_dataset(
        num_sequences=config['generation']['num_sequences'],
        stages_per_sequence=config['generation']['stages_per_sequence']
    )

    print("\nSimulation input files generated successfully!")
    print("Next step: Run gprMax on these input files")
    print("Command example: python -m gprMax data/simulations/seq_0000_stage_00.in")


if __name__ == '__main__':
    main()
