#!/bin/bash
# GPR simulation execution script

set -e

echo "=========================================="
echo "GPR Void Evolution Simulation"
echo "=========================================="

# Create directories
mkdir -p data/simulations
mkdir -p outputs

# Step 1: Generate gprMax input files
echo ""
echo "[1/2] Generating gprMax input files..."
python src/data_generation/gpr_simulator.py

# Step 2: Run gprMax
echo ""
echo "[2/2] Running gprMax simulations..."
echo "This may take a while..."

# Run in parallel using Python wrapper
python src/data_generation/run_gprmax.py \
    --input-dir data/simulations \
    --workers ${NUM_WORKERS:-4}

echo ""
echo "=========================================="
echo "Simulation completed!"
echo "Output files: data/simulations/*.out"
echo "=========================================="
