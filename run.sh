#!/bin/bash
# Start the Sentinel AI NIDS server
# Usage: ./run.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_PYTHON="/home/kuper/.conda/envs/nids/bin/python"

if [ ! -f "$CONDA_PYTHON" ]; then
    echo "❌ Conda env 'nids' not found. Create it with: conda create -n nids python=3.12"
    exit 1
fi

echo "🛡️  Starting Sentinel AI NIDS Server..."
echo "   Dashboard: http://localhost:8000"
echo "   Press Ctrl+C to stop"
echo ""

exec "$CONDA_PYTHON" "$SCRIPT_DIR/app.py"
