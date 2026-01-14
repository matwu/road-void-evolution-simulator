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
        self.surface_asphalt_thickness = road['surface_asphalt_thickness']
        self.base_asphalt_thickness = road['base_asphalt_thickness']
        self.upper_subbase_thickness = road['upper_subbase_thickness']
        self.lower_subbase_thickness = road['lower_subbase_thickness']
        self.subgrade_thickness = road['subgrade_thickness']

        # GPR parameters
        self.frequency = config['gpr']['frequency']  # Center frequency (MHz)
        self.time_window = config['gpr']['time_window']  # Time window (ns)
        self.spatial_resolution = config['gpr']['spatial_resolution']  # Spatial resolution (m)

        # Void parameter ranges
        void = config['void']
        self.void_initial_depth_range = void['initial_depth_range']
        self.void_initial_size_x_range = void['initial_size_x_range']
        self.void_initial_size_y_range = void['initial_size_y_range']
        self.void_initial_size_z_range = void['initial_size_z_range']
        self.void_growth_rate_range = void['growth_rate_range']
        self.void_upward_movement_range = void['upward_movement_range']

        # Materials
        self.materials = config['materials']

        # Domain
        domain = config['domain']
        self.domain_x = domain['size_x']
        self.domain_y = domain['size_y']
        self.domain_z = domain['size_z']

    def generate_void_parameters(self, stage: int, total_stages: int, sequence_seed: int = None) -> Dict:
        """
        Generate void time-evolution parameters

        Args:
            stage: Current stage (0 is initial, total_stages-1 is final)
            total_stages: Total number of stages
            sequence_seed: Seed to fix initial values per sequence

        Returns:
            Void parameter dictionary
        """
        # Progress (0.0 ~ 1.0)
        progress = stage / (total_stages - 1) if total_stages > 1 else 0

        # Generate consistent initial parameters within a sequence
        if sequence_seed is not None and stage == 0:
            np.random.seed(sequence_seed)

        # Initial position and size (get range from config)
        if stage == 0:
            # For stage 0, generate new initial values
            self._initial_depth = np.random.uniform(*self.void_initial_depth_range)
            self._initial_x_size = np.random.uniform(*self.void_initial_size_x_range)
            self._initial_y_size = np.random.uniform(*self.void_initial_size_y_range)
            self._initial_z_size = np.random.uniform(*self.void_initial_size_z_range)
            self._max_growth_rate = np.random.uniform(*self.void_growth_rate_range)
            self._max_upward_movement = np.random.uniform(*self.void_upward_movement_range)
            self._center_x = np.random.uniform(0.1, self.domain_x - 0.1)
            self._center_y = np.random.uniform(0.1, self.domain_y - 0.1)

        # Growth rate (simulate non-linear growth)
        growth_rate = 1.0 + progress ** 1.5 * (self._max_growth_rate - 1.0)

        # Upward movement (rising toward surface)
        upward_movement = progress * self._max_upward_movement

        return {
            'center_x': self._center_x,
            'center_y': self._center_y,
            'center_z': self._initial_depth - upward_movement,  # Rising
            'size_x': self._initial_x_size * growth_rate,  # x-direction growth
            'size_y': self._initial_y_size * growth_rate,  # y-direction growth
            'size_z': self._initial_z_size * growth_rate ** 0.8,  # z-direction grows more slowly
            'stage': stage,
            'progress': progress
        }

    def create_gpr_input_file(self, void_params: Dict, filename: str) -> str:
        """
        Generate gprMax input file (.in)

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

        # Calculate layer positions (top to bottom)
        # Z-coordinate is height from bottom
        z_top = domain_z
        z_surface_asphalt_bottom = z_top - self.surface_asphalt_thickness
        z_base_asphalt_bottom = z_surface_asphalt_bottom - self.base_asphalt_thickness
        z_upper_subbase_bottom = z_base_asphalt_bottom - self.upper_subbase_thickness
        z_lower_subbase_bottom = z_upper_subbase_bottom - self.lower_subbase_thickness
        z_subgrade_bottom = z_lower_subbase_bottom - self.subgrade_thickness

        content = f"""#title: Road void evolution stage {void_params['stage']}
#domain: {domain_x} {domain_y} {domain_z}
#dx_dy_dz: {dx} {dx} {dx}
#time_window: {self.time_window}e-9

#material: {self.materials['surface_asphalt']} 0.01 1 0 surface_asphalt
#material: {self.materials['base_asphalt']} 0.01 1 0 base_asphalt
#material: {self.materials['upper_subbase']} 0.02 1 0 upper_subbase
#material: {self.materials['lower_subbase']} 0.02 1 0 lower_subbase
#material: {self.materials['subgrade']} 0.05 1 0 subgrade
#material: {self.materials['void']} 0 1 0 void

#box: 0 0 {z_surface_asphalt_bottom} {domain_x} {domain_y} {z_top} surface_asphalt
#box: 0 0 {z_base_asphalt_bottom} {domain_x} {domain_y} {z_surface_asphalt_bottom} base_asphalt
#box: 0 0 {z_upper_subbase_bottom} {domain_x} {domain_y} {z_base_asphalt_bottom} upper_subbase
#box: 0 0 {z_lower_subbase_bottom} {domain_x} {domain_y} {z_upper_subbase_bottom} lower_subbase
#box: 0 0 {z_subgrade_bottom} {domain_x} {domain_y} {z_lower_subbase_bottom} subgrade

#cylinder: {void_params['center_x']} {void_params['center_y']} {void_params['center_z']} {void_params['center_x']} {void_params['center_y']} {void_params['center_z'] + void_params['size_z']} {void_params['size_x']/2} void

#waveform: ricker 1 {self.frequency}e6 my_ricker
#hertzian_dipole: z {domain_x/2} {domain_y/2} {z_top - 0.05} my_ricker

#rx: {domain_x/2} {domain_y/2} {z_top - 0.05}

#src_steps: {dx} 0 0
#rx_steps: {dx} 0 0

#geometry_view: 0 0 0 {domain_x} {domain_y} {domain_z} {dx} {dx} {dx} geometry_stage_{void_params['stage']} n
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
