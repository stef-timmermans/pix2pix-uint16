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


def _foreground_importance_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    background_percentile: float,
    min_importance: float,
    max_importance: float,
    importance_scale: float,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    Foreground-aware weighted L1 reconstruction loss.

    High-level behavior:
    - Inspect current target patch
    - Estimate background from the bottom x% of target intensities
    - Let pixels with intensity <= background have minimum importance weight
    - Let pixels increasingly above background have importance approaching maximum
    - Smooth and bound the weighting function so that background still has impact

    Benefits of approach:
    - No fixed absolute intensity tied to specific data
    - Importance defined relative to distance from patch-estimated background
    - Empty/background-only patches remain viable
    - Foreground importance is greatly increased but background is still considered

    Assumptions:
    - pred and target have shape [B, C, H, W]

    Args:
        pred (torch.Tensor):
            Predicted image tensor of shape [B, C, H, W].

        target (torch.Tensor):
            Ground-truth image tensor of shape [B, C, H, W].

        background_percentile (float):
            Percentage of darkest pixels in each patch used to estimate
            the local background reference.

        min_importance (float):
            Minimum weight assigned to background-like pixels.

        max_importance (float):
            Maximum weight assigned to strongly foreground-like pixels.

        importance_scale (float):
            Intensity distance above background in normalized tensor space
            at which importance approaches its upper range.

        gamma (float):
            Optional shape parameter controlling how quickly importance
            rises with distance above background.

    Returns:
        torch.Tensor:
            Scalar weighted L1 loss.
    """

    if pred.shape != target.shape:
        raise ValueError("pred and target must have identical shapes")

    if not (0.0 < background_percentile <= 100.0):
        raise ValueError("background_percentile must be in (0, 100]")

    if min_importance <= 0:
        raise ValueError("min_importance must be > 0")

    if max_importance < min_importance:
        raise ValueError("max_importance must be >= min_importance")

    if importance_scale <= 0:
        raise ValueError("importance_scale must be > 0")

    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    B, C, H, W = target.shape

    # Flatten spatial dims; operate per patch
    flat_target = target.view(B, C, -1)

    # Number of darkest pixels
    k = max(1, int(flat_target.shape[-1] * background_percentile / 100.0))

    # Get bottom x% intensities
    bottom_vals, _ = torch.topk(flat_target, k=k, dim=-1, largest=False)

    # Background estimate (median of bottom x%)
    background = bottom_vals.median(dim=-1, keepdim=True).values

    # Distance above background
    above_bg = (flat_target - background).clamp(min=0.0)

    # Smooth bounded importance curve in normalized tensor space
    scaled = (above_bg / importance_scale).clamp(min=0.0, max=1.0)
    scaled = scaled.pow(gamma)

    # Set weights
    weights = min_importance + (max_importance - min_importance) * scaled

    # Compute weighted L1
    flat_pred = pred.view(B, C, -1)
    l1_map = torch.abs(flat_pred - flat_target)

    # No weight normalization; preserves lower loss for background patches.
    # Loss = average importance-adjusted per-pixel reconstruction error.
    weighted = weights * l1_map
    return weighted.mean()


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
        importance_scale = (opt.importance_scale / max_val) * 2.0

        return _foreground_importance_l1(
            pred=pred,
            target=target,
            background_percentile=opt.background_percentile,
            min_importance=opt.min_importance,
            max_importance=opt.max_importance,
            importance_scale=importance_scale,
            gamma=opt.importance_gamma,
        )

    raise ValueError(f"Unknown loss_type: {opt.recon_loss}")
