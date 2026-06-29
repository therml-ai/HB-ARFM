from typing import List, Union
import torch
import torch.nn as nn

from .noise import (
    OpticalFlowNoise,
    SimpleGaussianNoise,
    SpatiallyCorrelatedNoise,
    GradientAwareNoise,
    VelocityScaledNoise,
    create_noise_model,
)


class LpLoss(nn.Module):
    """
    Lp loss on a tensor (b, n1, n2, ..., nd)
    Args:
        d (int): Number of dimensions to flatten from right
        p (int): Power of the norm
        reduce_dims (List[int]): Dimensions to reduce
        reductions (List[str]): Reductions to apply
    """
    def __init__(
            self,
            d: int = 1,
            p: int = 2,
            reduce_dims: Union[int, List[int]] = 0,
            reductions: Union[str, List[str]] = "sum"
        ):
        super().__init__()

        self.d = d
        self.p = p

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims

        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == "sum" or reductions == "mean"
                self.reductions = [reductions] * len(self.reduce_dims)
            else:
                for reduction in reductions:
                    assert reduction == "sum" or reduction == "mean"
                self.reductions = reductions

    def reduce_all(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduce the tensor along the specified dimensions
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Reduced tensor
        """
        for j, reduce_dim in enumerate(self.reduce_dims):
            if self.reductions[j] == "sum":
                x = torch.sum(x, dim=reduce_dim, keepdim=True)
            else:
                x = torch.mean(x, dim=reduce_dim, keepdim=True)
        return x

    def forward(
            self,
            y_pred: torch.Tensor,
            y: torch.Tensor
        ) -> torch.Tensor:
        """
        Args:
            y_pred (torch.Tensor): Predicted tensor
            y (torch.Tensor): Target tensor
        Returns:
            torch.Tensor: Lp loss
        """
        diff = torch.norm(
            torch.flatten(y_pred, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d),
            p=self.p,
            dim=-1,
            keepdim=False,
        )
        ynorm = torch.norm(
            torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False
        )

        diff = diff / ynorm

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()

        return diff
