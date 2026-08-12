"""Pinned product ``telos`` train/predict backend."""

from __future__ import annotations

import sys
from pathlib import Path

from telos_repro.backend.types import PredictRequest, TrainRequest
from telos_repro.paths import find_repo_root, load_paths, path_value


def ensure_telos_importable() -> Path:
    """
    Ensure the pinned Telos checkout is importable as ``telos``.

    Prefers an already-installed ``telos``; otherwise inserts ``{telos_checkout}/src``.
    """
    try:
        import telos  # noqa: F401

        return Path(telos.__file__).resolve().parent.parent  # type: ignore[name-defined]
    except ImportError:
        pass

    paths = load_paths(find_repo_root())
    checkout = path_value(paths, "telos_checkout")
    src = (checkout / "src").resolve()
    if not (src / "telos").is_dir():
        raise FileNotFoundError(
            f"Telos package not found under {src}. "
            "Install telos or set telos_checkout in configs/paths.yaml."
        )
    src_s = str(src)
    if src_s not in sys.path:
        sys.path.insert(0, src_s)
    import telos  # noqa: F401

    return src


class TelosBackend:
    name = "telos"

    def train(self, req: TrainRequest) -> int:
        ensure_telos_importable()
        from telos.commands.train import run_train
        from telos.config_models import TrainIO

        return run_train(
            TrainIO(
                bam=req.bam,
                gtf=req.gtf,
                outdir=req.outdir,
                ref_gtf=req.ref_gtf,
                tmap=req.tmap,
                config_file=req.config_file,
                gtf_pool=req.gtf_pool,
                tmap_pool=req.tmap_pool,
                stage1_no_parallel=req.stage1_no_parallel,
                stage1_n_workers=req.stage1_n_workers,
                split_policy=req.split_policy,
                n_jobs=req.n_jobs,
            )
        )

    def predict(self, req: PredictRequest) -> int:
        ensure_telos_importable()
        from telos.commands.predict import run_predict
        from telos.config_models import PredictIO

        return run_predict(
            PredictIO(
                bam=req.bam,
                gtf=req.gtf,
                outdir=req.outdir,
                model_dir=req.model_dir,
                config_file=req.config_file,
                backend=req.backend,
                min_score=req.min_score,
                stage1_no_parallel=req.stage1_no_parallel,
                stage1_n_workers=req.stage1_n_workers,
            )
        )
