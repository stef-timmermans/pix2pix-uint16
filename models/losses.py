import torch
import torch.nn.functional as F

from util.types import dtype_max


def _standard_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Standard mean L1 reconstruction loss.

    Args:
        pred (torch.Tensor):
            Predicted image tensor of shape [B, C, H, W].

        target (torch.Tensor):
            Ground-truth image tensor of shape [B, C, H, W].

    Returns:
        torch.Tensor:
            Scalar mean L1 loss.
    """
    return F.l1_loss(pred, target, reduction="mean")


def _foreground_aware_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    background_percentile: float,
    foreground_margin: float,
    fg_weight: float,
    bg_weight: float,
) -> torch.Tensor:
    """
    Foreground-aware reconstruction loss using explicit region separation.

    Key idea:
    ----------
    Instead of smoothly weighting all pixels (which gets dominated by background),
    we explicitly split pixels into:
        - foreground (important signal)
        - background (less important context)

    Then compute losses separately and weight them differently.

    Why this works:
    ---------------
    - Sparse targets (like fluorescence) are mostly background.
    - Standard L1 encourages predicting "nothing everywhere".
    - This loss forces the model to care about rare signal.

    Pipeline:
    ---------
    1. Estimate background level per patch using the lowest-intensity pixels.
    2. Define foreground as pixels sufficiently above background.
    3. Compute L1 separately for foreground and background.
    4. Combine with strong foreground weighting.

    This avoids:
    -----------
    - loss dilution from large background regions
    - trivial solutions (predicting near-zero everywhere)
    """

    if pred.shape != target.shape:
        raise ValueError("pred and target must have identical shapes")

    if not (0.0 < background_percentile <= 100.0):
        raise ValueError("background_percentile must be in (0, 100]")

    if foreground_margin < 0:
        raise ValueError("foreground_margin must be >= 0")

    if fg_weight <= 0 or bg_weight < 0:
        raise ValueError("fg_weight must be > 0 and bg_weight must be >= 0")

    B, C, H, W = target.shape

    # Flatten spatial dims; operate per patch
    flat_target = target.view(B, C, -1)

    # Number of darkest pixels used to estimate background
    k = max(1, int(flat_target.shape[-1] * background_percentile / 100.0))

    # Get bottom x% intensities (assumed to be background)
    bottom_vals, _ = torch.topk(flat_target, k=k, dim=-1, largest=False)

    # Estimate background as median of darkest pixels
    # (robust to noise and outliers)
    background = bottom_vals.median(dim=-1, keepdim=True).values

    # Reshape back to image shape
    background = background.view(B, C, 1, 1)

    # Define foreground mask:
    # pixels significantly above estimated background
    fg_mask = target > (background + foreground_margin)

    # Everything else is background
    bg_mask = ~fg_mask

    # Absolute error per pixel
    abs_err = torch.abs(pred - target)

    # Count pixels in each region (avoid division by zero)
    fg_count = fg_mask.sum().clamp_min(1)
    bg_count = bg_mask.sum().clamp_min(1)

    # Mean L1 over foreground pixels only
    fg_l1 = abs_err[fg_mask].sum() / fg_count

    # Mean L1 over background pixels only
    bg_l1 = abs_err[bg_mask].sum() / bg_count

    # Combine losses with explicit weighting
    # Foreground should dominate learning signal
    return (fg_weight * fg_l1) + (bg_weight * bg_l1)


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    opt,
) -> torch.Tensor:
    """
    Dispatcher for reconstruction loss types.

    Args:
        pred (torch.Tensor):
            Predicted image tensor.

        target (torch.Tensor):
            Ground-truth image tensor.

        opt:
            Parsed options object containing reconstruction-loss configuration.

    Returns:
        torch.Tensor:
            Scalar reconstruction loss.
    """
    # Convert appropriate hyperparameters to the normalized tensor space [-1, 1].
    max_val = dtype_max(opt.dtype)

    if opt.recon_loss == "l1":
        return _standard_l1(pred, target)

    if opt.recon_loss == "foreground_aware":
        # Convert margin from raw dtype space (e.g. uint16) to normalized [-1, 1]
        foreground_margin = (opt.foreground_margin / max_val) * 2.0

        return _foreground_aware_l1(
            pred=pred,
            target=target,
            background_percentile=opt.background_percentile,
            foreground_margin=foreground_margin,
            fg_weight=opt.fg_weight,
            bg_weight=opt.bg_weight,
        )

    raise ValueError(f"Unknown loss_type: {opt.recon_loss}")
