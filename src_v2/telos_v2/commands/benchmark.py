from __future__ import annotations

from telos_v2.config_models import BenchmarkIO


def run_benchmark(cfg: BenchmarkIO) -> int:
    """
    Benchmark command scaffold.

    Planned behavior:
    - parse experiment config
    - run train/predict sweeps (including cross-annotation evaluation)
    - produce tables/plots and reproducibility manifest
    """
    print("[telos_v2] benchmark scaffold")
    print(f"  config={cfg.config}")
    print(f"  outdir={cfg.outdir}")
    return 0
