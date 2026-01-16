#!/bin/bash
# Quick start script

set -e

echo "=========================================="
echo "GPR Void Detection AI - Quick Start"
echo "=========================================="

# 1. Build Docker environment
echo ""
echo "[Step 1/4] Building Docker environment..."
docker-compose build

# 2. Start container
echo ""
echo "[Step 2/4] Starting container..."
docker-compose up -d

# 3. Create directories
echo ""
echo "[Step 3/4] Creating directories..."
docker-compose exec -T road-void-evolution-simulator mkdir -p data/simulations
docker-compose exec -T road-void-evolution-simulator mkdir -p outputs

# 4. Generate simulation data
echo ""
echo "[Step 4/4] Generating simulation data..."
docker-compose exec -T road-void-evolution-simulator python src/data_generation/gpr_simulator.py

echo ""
echo "=========================================="
echo "Setup completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run simulations:"
echo "     docker-compose exec road-void-evolution-simulator bash scripts/run_simulation.sh"
echo ""
echo "To enter the container:"
echo "  docker-compose exec road-void-evolution-simulator /bin/bash"
echo ""
