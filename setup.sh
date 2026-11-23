#!/usr/bin/env bash
# Set up the Simulation Lab's local GreenLight-Gym2 service.
#   ./setup.sh            simulation only (gl-gym + FastAPI)
#   ./setup.sh --train    also Gym2's training extras (PyTorch, Stable-Baselines3, W&B, ...)
set -euo pipefail
cd "$(dirname "$0")"

GYM2_REPO="https://github.com/BartvLaatum/GreenLight-Gym2"
GYM2_COMMIT="da8b22ca68ebb74e34cd5c7881b90c2b14e41ced"   # the commit the Lab was built and tested against
PYTHON="${PYTHON:-$(command -v python3.12 || command -v python3.11 || command -v python3)}"

"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 11), "GreenLight-Gym2 needs Python 3.11 or newer"' \
  || { echo "Python 3.11+ is required (e.g. brew install python@3.12), then PYTHON=python3.12 ./setup.sh"; exit 1; }

if [ ! -d vendor/GreenLight-Gym2/.git ]; then
  mkdir -p vendor
  git clone "$GYM2_REPO" vendor/GreenLight-Gym2
fi
git -C vendor/GreenLight-Gym2 fetch --quiet origin "$GYM2_COMMIT" 2>/dev/null || true
git -C vendor/GreenLight-Gym2 checkout --quiet "$GYM2_COMMIT"

[ -d .venv ] || "$PYTHON" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
if [ "${1:-}" = "--train" ]; then
  .venv/bin/pip install --quiet -e "vendor/GreenLight-Gym2[train]"
else
  .venv/bin/pip install --quiet -e "vendor/GreenLight-Gym2"
fi
.venv/bin/pip install --quiet -r requirements.txt

echo "Ready. Start the service with:"
echo "  .venv/bin/python -m uvicorn service.app:app --host 127.0.0.1 --port 8100"
