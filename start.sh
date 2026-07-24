#!/usr/bin/env bash
#
# Bootstrap and run the contract-reviewer backend.
#
# Ensures dependencies are installed (uv sync) and forwards any arguments
# to the CLI. Examples:
#
#   ./start.sh ingest path/to/contract.pdf
#   ./start.sh query my_collection --top-k 5
#
# Run with no arguments to see the CLI help.

set -euo pipefail

# Resolve the repo root (directory containing this script) so the script
# works regardless of the caller's current directory.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: 'uv' is not installed. Install it from https://docs.astral.sh/uv/ and retry." >&2
  exit 1
fi

cd "$BACKEND_DIR"

# Ignore any venv the caller has activated (e.g. an old root .venv) so uv
# consistently targets backend/.venv and doesn't emit a mismatch warning.
unset VIRTUAL_ENV

# Install/refresh dependencies into backend/.venv (fast no-op when up to date).
echo "==> Syncing dependencies (uv sync)"
uv sync

if [ "$#" -eq 0 ]; then
  echo "==> No command given. Showing CLI help."
  exec uv run python app.py --help
fi

echo "==> Running: python app.py $*"
exec uv run python app.py "$@"
