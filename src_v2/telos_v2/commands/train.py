from __future__ import annotations

from telos_v2.config_models import TrainIO


def run_train(cfg: TrainIO) -> int:
    """
    Train command scaffold.

    Planned behavior:
    - preflight input validation
    - Stage I feature build and site-model training
    - Stage II transcript-model training
    - write model artifacts and run manifest
    """
    print("[telos_v2] train scaffold")
    print(f"  bam={cfg.bam}")
    print(f"  gtf={cfg.gtf}")
    print(f"  ref_gtf={cfg.ref_gtf}")
    print(f"  tmap={cfg.tmap}")
    print(f"  outdir={cfg.outdir}")
    return 0
