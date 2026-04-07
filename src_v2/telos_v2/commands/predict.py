from __future__ import annotations

from telos_v2.config_models import PredictIO


def run_predict(cfg: PredictIO) -> int:
    """
    Predict command scaffold.

    Planned behavior:
    - load pretrained Stage I + Stage II models
    - score candidate sites
    - score transcripts
    - emit ranked transcripts and filtered GTF
    """
    print("[telos_v2] predict scaffold")
    print(f"  bam={cfg.bam}")
    print(f"  gtf={cfg.gtf}")
    print(f"  model_dir={cfg.model_dir}")
    print(f"  outdir={cfg.outdir}")
    return 0
