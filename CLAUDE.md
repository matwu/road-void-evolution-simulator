# CLAUDE.MD - Project Context for AI Assistants

## Project Overview

**Road Void Evolution Simulator** - A GPR (Ground Penetrating Radar) training data generation tool that simulates temporal evolution of road subsurface voids using gprMax.

### Purpose
Generate high-quality training data for vehicle-mounted compact GPR systems to detect and track road subsurface voids as they evolve over time (growth and upward movement).

### Key Technologies
- **gprMax**: Electromagnetic wave simulation engine
- **Python**: Data processing and visualization
- **Docker**: Containerized execution environment
- **HDF5**: Binary simulation output format
- **CSV**: Machine learning-ready data format

## Project Structure

```
.
├── config/                          # Configuration files
│   └── simulation_config.yaml       # Simulation parameters
├── data/                            # Data directory
│   └── simulations/                 # Simulation outputs
│       ├── *.in                     # gprMax input files
│       ├── *.out                    # gprMax output files (HDF5)
│       ├── metadata.yaml            # Simulation metadata
│       ├── csv/                     # CSV exports
│       └── plots/                   # Visualization outputs
├── outputs/                         # Generated outputs
├── scripts/                         # Shell scripts
│   └── run_simulation.sh            # Main simulation runner
├── src/                             # Source code
│   ├── data_generation/             # Data generation modules
│   │   ├── gpr_simulator.py         # GPR simulation logic
│   │   ├── run_gprmax.py            # gprMax runner
│   │   └── export_to_csv.py         # CSV export utility
│   └── visualization/               # Visualization modules
│       └── plot_gpr_output.py       # Plotting utility
├── Dockerfile                       # Docker image definition
├── docker-compose.yml               # Docker compose config
├── patch_gprmax.py                  # gprMax patching script
├── requirements.txt                 # Python dependencies
├── README.md                        # User-facing documentation
└── USAGE_GUIDE.md                   # Detailed usage instructions
```

## Core Concepts

### Simulation Data
- **Sequences**: Independent simulation runs (seq_XXXX)
- **Stages**: Time steps within a sequence showing void evolution (stage_YY)
- **Components**: 6 electromagnetic field components (Ex, Ey, Ez, Hx, Hy, Hz)
- **Time-series**: Each simulation produces temporal waveform data

### Void Evolution
The simulator models how subsurface voids change over time:
1. **Depth Rise**: Voids move upward toward the road surface
2. **Expansion**: Voids grow in x/y directions
3. **Temporal Tracking**: Multiple stages capture the evolution process

### Data Format
- **Input**: `.in` files (gprMax input scripts with detailed inline comments)
  - Includes parameter explanations and coordinate system documentation
  - Self-documenting format for easy understanding
- **Simulation Output**: `.out` files (HDF5 binary format)
- **ML-Ready**: `.csv` files (time_ns, Ex, Ey, Ez, Hx, Hy, Hz)
- **Visualization**: PNG plots (A-scan, B-scan, all components)

### Scan Types
- **A-scan**: Single-position time-domain waveform (1D: time)
- **B-scan**: Cross-sectional image from multiple trace positions (2D: distance x time)
  - Transmitter and receiver move together along x-axis
  - Creates subsurface profile image
  - Number of traces controlled by `num_traces` parameter

## Development Environment

### Docker Setup
```bash
# Build and start
docker-compose build
docker-compose up -d

# Enter container
docker-compose exec gprmax-docker /bin/bash
```

### Container Environment
- Base: Ubuntu with gprMax
- Working directory: /workspace
- Mounted volumes: Local project directory to container

## Common Tasks

### Running Simulations
```bash
# Inside container
bash scripts/run_simulation.sh

# With custom worker count
NUM_WORKERS=8 bash scripts/run_simulation.sh
```

### CSV Export
```bash
# Single file
python src/data_generation/export_to_csv.py data/simulations/seq_0000_stage_00.out --output output.csv

# Batch processing
python src/data_generation/export_to_csv.py data/simulations --output data/simulations/csv --mode batch

# Sequence combination
python src/data_generation/export_to_csv.py data/simulations --output combined.csv --mode sequence --component Ez --pattern "seq_0000_stage_*.out"
```

### Visualization
```bash
# All components
python src/visualization/plot_gpr_output.py data/simulations/seq_0000_stage_00.out --mode all --output plots/

# A-scan (time-domain waveform)
python src/visualization/plot_gpr_output.py file.out --mode ascan --component Ez --output ascan.png

# B-scan (spatial-temporal image)
python src/visualization/plot_gpr_output.py data/simulations --mode bscan --component Ez --output bscan.png
```

## Configuration

### simulation_config.yaml
Key parameters to adjust:
- `num_sequences`: Number of independent simulations
- `stages_per_sequence`: Evolution steps per sequence
- `road.air_thickness`: Air layer thickness above road surface (m)
- `road.*_thickness`: Road layer thicknesses (surface asphalt, base asphalt, subbase, subgrade)
- `gpr.frequency`: GPR antenna frequency (MHz)
- `gpr.time_window`: Simulation time window (ns)
- `gpr.spatial_resolution`: Spatial discretization (m)
- `gpr.num_traces`: Number of B-scan traces (measurement positions)
- `gpr.scan_start_x`: B-scan start position along x-axis (m)
- `gpr.scan_end_x`: B-scan end position along x-axis (m)
- `domain.size_x`, `domain.size_y`: Horizontal domain dimensions (m)
- **Note**: `domain.size_z` is automatically calculated from road layer thicknesses

## Important Notes

### When Working on This Project

1. **Always use Docker**: gprMax has specific dependencies, run everything in container
2. **HDF5 format**: `.out` files are binary HDF5, not human-readable
3. **File naming**: Pattern is `seq_XXXX_stage_YY.{in,out}` with zero-padded numbers
4. **Memory usage**: Reduce NUM_WORKERS if running out of memory
5. **Component names**: Use Ex, Ey, Ez, Hx, Hy, Hz (case-sensitive)
6. **Domain height**: Automatically calculated from air + road layer thicknesses (no manual setting needed)
7. **Sensor position**: GPR antenna positioned at road surface (logical z=0)
8. **Coordinate system**: `.in` files include detailed comments explaining the coordinate system and parameter meanings

### Code Patterns

- Python scripts use argparse for CLI arguments
- Scripts expect to be run from project root
- HDF5 reading uses h5py library
- Visualization uses matplotlib
- Path handling uses pathlib.Path

### Common Issues

1. **Out of Memory**: Reduce NUM_WORKERS or reduce spatial resolution
2. **Missing gprMax**: Ensure running inside Docker container
3. **HDF5 errors**: Check that .out files exist and are complete
4. **Plot rendering**: May need backend configuration for headless environments

## Testing and Validation

- Verify simulation outputs have expected time steps
- Check CSV exports match HDF5 data
- Validate void parameters are within physical bounds
- Ensure plots show expected GPR characteristics (hyperbolic reflections)

## Future Considerations

- Parallel processing optimization
- Additional material types (soil, concrete variations)
- Real-time visualization during simulation
- Automated quality checks for generated data
- Integration with ML training pipelines

## References

- gprMax documentation: https://www.gprmax.com/
- GPR principles: Ground-penetrating radar fundamentals
- Data format: HDF5 file format specification
