"""
Configurable loss functions for flow matching models.

Supports individual losses (mse, l1, huber, relative_l1, relative_l2) and
weighted hybrid combinations specified via YAML config.

Usage in YAML config:
    # Single loss (default):
    loss_type: mse

    # Huber loss (L2 for small errors, L1 for large):
    loss_type: huber
    loss_delta: 1.0   # transition threshold

    # Hybrid loss:
    loss_type: hybrid
    loss_weights:
      l1: 0.8
      mse: 0.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def relative_l1_loss(pred: torch.Tensor, target: torch.Tensor,
                     eps: float = 0.01) -> torch.Tensor:
    """Element-wise relative L1: |pred - target| / (|target| + eps)."""
    return (torch.abs(pred - target) / (torch.abs(target) + eps)).mean()


def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor,
                     eps: float = 0.01) -> torch.Tensor:
    """Element-wise relative L2: (pred - target)^2 / (target^2 + eps)."""
    return ((pred - target).pow(2) / (target.pow(2) + eps)).mean()


_LOSS_REGISTRY = {
    'mse': lambda: nn.MSELoss(),
    'l1': lambda: nn.L1Loss(),
    'huber': lambda delta=1.0: nn.HuberLoss(delta=delta),
    'relative_l1': lambda eps=0.01: (lambda p, t: relative_l1_loss(p, t, eps)),
    'relative_l2': lambda eps=0.01: (lambda p, t: relative_l2_loss(p, t, eps)),
}


class HybridLoss(nn.Module):
    """Weighted combination of multiple loss functions.

    Args:
        weights: dict mapping loss name -> weight, e.g. {'l1': 0.8, 'mse': 0.2}
        eps: epsilon for relative losses
    """

    def __init__(self, weights: dict, eps: float = 0.01, delta: float = 1.0):
        super().__init__()
        self.components = {}
        self.weights = {}
        for name, w in weights.items():
            if w <= 0:
                continue
            if name in ('mse', 'l1'):
                self.components[name] = _LOSS_REGISTRY[name]()
            elif name == 'huber':
                self.components[name] = _LOSS_REGISTRY[name](delta)
            elif name in ('relative_l1', 'relative_l2'):
                self.components[name] = _LOSS_REGISTRY[name](eps)
            else:
                raise ValueError(f"Unknown loss component '{name}'. "
                                 f"Choose from: {list(_LOSS_REGISTRY.keys())}")
            self.weights[name] = w

        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            print(f"   ⚠️  Loss weights sum to {total:.3f}, normalizing to 1.0")
            for k in self.weights:
                self.weights[k] /= total

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for name, fn in self.components.items():
            total = total + self.weights[name] * fn(pred, target)
        return total

    def __repr__(self):
        parts = [f"{w:.2f}*{n}" for n, w in self.weights.items()]
        return f"HybridLoss({' + '.join(parts)})"


def build_loss_fn(model_cfg) -> nn.Module:
    """Build a loss function from model config.

    Config keys used:
        loss_type: 'mse' | 'l1' | 'huber' | 'relative_l1' | 'relative_l2' | 'hybrid'
                   (default: 'mse')
        loss_weights: dict of {loss_name: weight} (only used when loss_type='hybrid')
        loss_eps: epsilon for relative losses (default: 0.01)
        loss_delta: delta threshold for Huber loss (default: 1.0)

    Returns a callable(pred, target) -> scalar loss.
    """
    loss_type = model_cfg.get('loss_type', 'mse')
    eps = model_cfg.get('loss_eps', 0.01)
    delta = model_cfg.get('loss_delta', 1.0)

    if loss_type == 'hybrid':
        weights = model_cfg.get('loss_weights', {'l1': 0.8, 'mse': 0.2})
        weights = dict(weights)
        loss_fn = HybridLoss(weights, eps=eps, delta=delta)
        print(f"   Loss: {loss_fn}")
    elif loss_type in ('mse', 'l1'):
        loss_fn = _LOSS_REGISTRY[loss_type]()
        print(f"   Loss: {loss_type.upper()}")
    elif loss_type == 'huber':
        loss_fn = _LOSS_REGISTRY['huber'](delta)
        print(f"   Loss: Huber (delta={delta})")
    elif loss_type in ('relative_l1', 'relative_l2'):
        loss_fn = _LOSS_REGISTRY[loss_type](eps)
        print(f"   Loss: {loss_type} (eps={eps})")
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'. "
                         f"Choose from: mse, l1, huber, relative_l1, relative_l2, hybrid")

    return loss_fn
