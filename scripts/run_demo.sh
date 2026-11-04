#!/usr/bin/env bash
# ==============================================================================
# scripts/run_demo.sh
# ==============================================================================
# One-command production launch for ResumeIQ (React + FastAPI).
# - Sets segfault-mitigation environment variables for PyTorch / macOS.
# - Builds the production frontend dist bundle if missing.
# - Warms up neural models (SBERT + DistilBERT + MLP).
# - Launches FastAPI with Uvicorn serving both REST API and React static SPA.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "========================================================================"
echo "ResumeIQ — Starting Production Demo Server"
echo "========================================================================"

# 1. Segfault mitigation environment variables
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 2. Check / build frontend dist if missing
if [ ! -d "frontend/dist" ] || [ ! -f "frontend/dist/index.html" ]; then
    echo "[1/3] Building React production frontend (frontend/dist) ..."
    npm --prefix frontend run build
else
    echo "[1/3] React frontend production build verified in frontend/dist."
fi

# 3. Warm up neural models in-memory
echo "[2/3] Warming up pipeline models in-memory ..."
./venv/bin/python -c "from app.pipeline import warm_up; warm_up()"

# 4. Launch FastAPI server (serving frontend/dist at root and /api/* endpoints)
echo "[3/3] Launching FastAPI backend on http://localhost:8000 ..."
echo "------------------------------------------------------------------------"
echo "  ResumeIQ Web App:  http://localhost:8000"
echo "  Interactive Docs:  http://localhost:8000/docs"
echo "  Health Check:      http://localhost:8000/health"
echo "------------------------------------------------------------------------"
echo "Press Ctrl+C to stop the server."

exec ./venv/bin/uvicorn app.api.main:app --host 127.0.0.1 --port 8000
