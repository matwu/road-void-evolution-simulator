# Usage Guide

## Setup

```bash
# Build Docker environment
docker-compose build
docker-compose up -d

# Enter container
docker-compose exec road-void-evolution-simulator /bin/bash
```

## Data Generation Flow

### 1. Run Simulation

```bash
# Automatic execution script
bash scripts/run_simulation.sh

# Specify number of workers
NUM_WORKERS=8 bash scripts/run_simulation.sh
```

### 2. CSV Export

#### Single File

```bash
python src/data_generation/export_to_csv.py \
    data/simulations/seq_0000_stage_00.out \
    --output data/simulations/seq_0000_stage_00.csv
```

#### Batch Conversion

```bash
python src/data_generation/export_to_csv.py \
    data/simulations \
    --output data/simulations/csv \
    --mode batch
```

#### Sequence Combination

```bash
# Combine all stages of seq_0000 into one CSV
python src/data_generation/export_to_csv.py \
    data/simulations \
    --output data/simulations/seq_0000_combined.csv \
    --mode sequence \
    --component Ez \
    --pattern "seq_0000_stage_*.out"
```

### 3. Visualization

#### All Components Plot

```bash
python src/visualization/plot_gpr_output.py \
    data/simulations/seq_0000_stage_00.out \
    --mode all \
    --output data/simulations/plots
```

#### A-scan

```bash
python src/visualization/plot_gpr_output.py \
    data/simulations/seq_0000_stage_00.out \
    --mode ascan \
    --component Ez \
    --output outputs/ascan.png
```

#### B-scan

```bash
python src/visualization/plot_gpr_output.py \
    data/simulations \
    --mode bscan \
    --component Ez \
    --output outputs/bscan.png
```

## Configuration Customization

Edit `config/simulation_config.yaml`:

```yaml
# Simulation parameters
generation:
  num_sequences: 100         # Number of sequences
  stages_per_sequence: 5     # Number of stages per sequence

# Road parameters
road:
  surface_asphalt_thickness: 0.04  # Surface asphalt thickness (m)
  base_asphalt_thickness: 0.06     # Base asphalt thickness (m)

# GPR parameters
gpr:
  frequency: 600              # GPR frequency (MHz)
  time_window: 2000           # Time window (ns)
  spatial_resolution: 0.005   # Spatial resolution (m)
```

## Output Files

### Directory Structure

```
data/simulations/
├── seq_XXXX_stage_YY.in      # gprMax input
├── seq_XXXX_stage_YY.out     # gprMax output (HDF5)
├── metadata.yaml             # Metadata
├── csv/                      # CSV output
│   └── seq_XXXX_stage_YY.csv
└── plots/                    # Visualization images
    └── seq_XXXX_stage_YY_all_components.png
```

### CSV Format

```csv
time_ns,Ex,Ey,Ez,Hx,Hy,Hz
0.0,0.0,0.0,0.0,0.0,0.0,0.0
0.019,1.23,2.34,3.45,0.12,0.23,0.34
...
```

## Troubleshooting

### Out of Memory

Reduce number of workers:
```bash
NUM_WORKERS=2 bash scripts/run_simulation.sh
```

### Docker Rebuild

```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```
