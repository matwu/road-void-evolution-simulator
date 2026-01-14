# GPR Training Data Generator

Time-series simulation data generation tool for road subsurface voids using gprMax

## Overview

This project is a tool for generating training data for vehicle-mounted compact GPR (Ground Penetrating Radar).
It uses the gprMax simulator to simulate the temporal changes (growth and upward movement) of road subsurface voids, generating CSV data and visualization images for machine learning.

### Key Features

- **gprMax Simulation**: Simulate void temporal evolution
- **CSV Export**: Export data in CSV format for machine learning
- **Visualization**: Output GPR data as images
- **Docker Environment**: Fully containerized execution environment

### Generated Data

- **Void Temporal Changes**: Depth rise and x/y direction expansion
- **GPR Waveform Data**: 6 components of electric field (Ex, Ey, Ez) and magnetic field (Hx, Hy, Hz)
- **CSV Format**: Output as time-series data

## Setup

### Requirements

- Docker
- Docker Compose

### Installation

```bash
# Build Docker image
docker-compose build

# Start container
docker-compose up -d

# Enter container
docker-compose exec gprmax /bin/bash
```

## Usage

### Step 1: Generate Simulation Data

```bash
# Run inside container
bash scripts/run_simulation.sh
```

### Step 2: Export to CSV Format

```bash
# Batch export
python src/data_generation/export_to_csv.py \
    data/simulations \
    --output data/simulations/csv \
    --mode batch
```

### Step 3: Visualization

```bash
python src/visualization/plot_gpr_output.py \
    data/simulations/seq_0000_stage_00.out \
    --mode all \
    --output data/simulations/plots
```

For details, see `SETUP_GUIDE.md`.
