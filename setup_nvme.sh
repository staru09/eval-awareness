#!/usr/bin/env bash
# Build the venv on the local NVMe disk. /nvme is wiped when the VM stops,
# so run this again after every start. Code stays on the file share.
set -euo pipefail

cd "$(dirname "$0")"

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

# Cache on the same disk as the venv so uv can hardlink instead of copying.
export UV_CACHE_DIR=/nvme/uv-cache
export UV_PROJECT_ENVIRONMENT=/nvme/venvs/evalaware

uv sync --extra api

echo
echo "Done. Add these to ~/.bashrc so 'uv run' uses this venv:"
echo "  export UV_CACHE_DIR=$UV_CACHE_DIR"
echo "  export UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT"
echo "Or activate it directly: source $UV_PROJECT_ENVIRONMENT/bin/activate"
