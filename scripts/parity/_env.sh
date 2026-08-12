# Shared env bootstrap for scripts/parity/*.sh
# shellcheck shell=bash
#
# Optional overrides (env or scripts/parity/local.env — gitignored):
#   TELOS_SRC           product Telos src/   (default: ../Telos/src next to this repo)
#   TELOS_BUNDLES_ROOT  bundle tree         (default: from paths.yaml or ./data/bundles)
#   GFFCOMPARE          gffcompare binary   (default: from paths.yaml or "gffcompare")
#   REF_RUNS_ROOT       reference runs dir  (optional; for compare scripts)
#   CONDA_ENV           activate this conda env if set

_PARITY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_PARITY_DIR}/../.." && pwd)"

if [[ -f "${_PARITY_DIR}/local.env" ]]; then
  # shellcheck disable=SC1091
  source "${_PARITY_DIR}/local.env"
fi

_PATHS_TELOS_SRC=""
_PATHS_BUNDLES=""
_PATHS_GFFCOMPARE=""
_PATHS_REF_RUNS=""

if [[ -d "${REPO_ROOT}/src/telos_repro" ]]; then
  # Prefer paths.yaml values when present (falls back to paths.example.yaml).
  eval "$(
    cd "${REPO_ROOT}" && PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python - <<'PY'
from __future__ import annotations
import shlex
from pathlib import Path
try:
    from telos_repro.paths import find_repo_root, load_paths, path_value
except Exception:
    raise SystemExit(0)
root = find_repo_root()
paths = load_paths(root)

def emit(name: str, val: str) -> None:
    print(f"{name}={shlex.quote(val)}")

if "telos_checkout" in paths:
    src = path_value(paths, "telos_checkout") / "src"
    if src.is_dir():
        emit("_PATHS_TELOS_SRC", str(src))
if "bundles_root" in paths:
    emit("_PATHS_BUNDLES", str(path_value(paths, "bundles_root")))
if "gffcompare_bin" in paths and paths["gffcompare_bin"]:
    emit("_PATHS_GFFCOMPARE", str(paths["gffcompare_bin"]))
if "goldens_runs_root" in paths:
    emit("_PATHS_REF_RUNS", str(path_value(paths, "goldens_runs_root")))
PY
  )" 2>/dev/null || true
fi

TELOS_SRC="${TELOS_SRC:-${_PATHS_TELOS_SRC:-${REPO_ROOT}/../Telos/src}}"
TELOS_BUNDLES_ROOT="${TELOS_BUNDLES_ROOT:-${_PATHS_BUNDLES:-${REPO_ROOT}/data/bundles}}"
GFFCOMPARE="${GFFCOMPARE:-${_PATHS_GFFCOMPARE:-gffcompare}}"
REF_RUNS_ROOT="${REF_RUNS_ROOT:-${_PATHS_REF_RUNS:-}}"

export REPO_ROOT TELOS_SRC TELOS_BUNDLES_ROOT GFFCOMPARE REF_RUNS_ROOT
export TELOS_REPRO_BACKEND="${TELOS_REPRO_BACKEND:-telos}"
export PYTHONPATH="${REPO_ROOT}/src:${TELOS_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${CONDA_ENV:-}" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

cd "${REPO_ROOT}"
