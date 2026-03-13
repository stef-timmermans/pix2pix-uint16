import torch
import torch.nn.functional as F


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
    gamma: float,
    eps: float = 1e-8,
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
    - Importance defined based on patch's own distribution; no messy I/O
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

        gamma (float):
            Shape parameter controlling how quickly importance increases
            as intensity rises above the background reference.

        eps (float):
            Small constant used to prevent division-by-zero during normalization.

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

    # Normalize per patch to stabilize scale (range [0, 1])
    max_above_bg = above_bg.amax(dim=-1, keepdim=True).clamp(min=eps)
    norm_above_bg = above_bg / max_above_bg

    # Smooth importance curve
    scaled = norm_above_bg.pow(gamma)
    weights = min_importance + (max_importance - min_importance) * scaled

    # Compute weighted L1
    flat_pred = pred.view(B, C, -1)
    l1_map = torch.abs(flat_pred - flat_target)

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

    if opt.recon_loss == "l1":
        return _standard_l1(pred, target)

    if opt.recon_loss == "foreground_aware":
        return _foreground_importance_l1(
            pred=pred,
            target=target,
            background_percentile=opt.background_percentile,
            min_importance=opt.min_importance,
            max_importance=opt.max_importance,
            gamma=opt.importance_gamma,
        )

    raise ValueError(f"Unknown loss_type: {opt.recon_loss}")
