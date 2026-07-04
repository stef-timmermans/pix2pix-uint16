import os
from netrc import netrc, NetrcParseError
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import torch.distributed as dist

from util import util

try:
    import wandb
except ImportError:
    wandb = None


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _wandb_host() -> str:
    base_url = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")
    parsed = urlparse(base_url)
    return parsed.netloc or parsed.path or "api.wandb.ai"


def has_wandb_credentials() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True

    try:
        auth = netrc().authenticators(_wandb_host())
    except (FileNotFoundError, NetrcParseError):
        auth = None

    return auth is not None

# noinspection PyUnreachableCode
def init_wandb_run(opt, *, job_type: str, run_name: Optional[str] = None):
    if not getattr(opt, "use_wandb", False):
        return None

    os.environ["WANDB_MODE"] = getattr(opt, "wandb_mode", "online")

    if wandb is None:
        raise ImportError("wandb is not installed. Install it or run without --use_wandb.")

    if not has_wandb_credentials():
        raise RuntimeError(
            "W&B is enabled for this run, but no credentials were detected. "
            "Set WANDB_API_KEY or run wandb login before launching."
        )

    if not is_main_process():
        return None

    os.environ.setdefault("WANDB_PROJECT", getattr(opt, "wandb_project_name", "pix2pix-uint16"))
    if getattr(opt, "wandb_entity", None):
        os.environ.setdefault("WANDB_ENTITY", opt.wandb_entity)

    run = (
        wandb.init(
            project=getattr(opt, "wandb_project_name", "pix2pix-uint16"),
            entity=getattr(opt, "wandb_entity", None),
            name=run_name or opt.name,
            config=vars(opt),
            job_type=job_type,
        )
        if not wandb.run
        else wandb.run
    )
    run._label(repo="pix2pix-uint16")
    return run


def log_visuals(run, visuals, *, step: int, prefix: str, output_imtype=np.uint8):
    if run is None:
        return

    payload = {}
    for label, image in visuals.items():
        image_numpy = util.tensor2im(image, imtype=output_imtype)
        payload[f"{prefix}/{label}"] = wandb.Image(
            image_numpy,
            caption=f"{label} - step {step}",
        )

    if payload:
        run.log(payload, step=step)


def log_metrics(run, metrics: dict, *, step: int):
    if run is None or not metrics:
        return
    run.log(metrics, step=step)


def update_config(run, values: dict):
    if run is None or not values:
        return
    run.config.update(values, allow_val_change=True)


def update_summary(run, metrics: dict):
    if run is None or not metrics:
        return
    for key, value in metrics.items():
        run.summary[key] = value


def finish_run(run):
    if run is not None:
        run.finish()
