"""
Noise models for simulating optical flow uncertainty.

This module provides noise injection utilities to simulate the type of errors
that arise when bubble position (SDF) and interface velocity are estimated
from optical flow on boiling videos, rather than from noise-free simulations.

Optical flow errors typically have these characteristics:
1. Spatially correlated errors: neighboring pixels have similar errors
2. Edge/gradient uncertainty: higher error at object boundaries
3. Motion-dependent noise: error scales with velocity magnitude
4. Heteroscedastic noise: variance varies spatially

Classes:
    OpticalFlowNoise: Main noise model combining multiple noise sources (complex)
    SimpleGaussianNoise: Simple i.i.d. Gaussian noise for baseline comparisons
    SpatiallyCorrelatedNoise: Smooth random fields via Gaussian filtering
    GradientAwareNoise: Higher noise at edges (for SDF boundaries)
    VelocityScaledNoise: Noise proportional to velocity magnitude

Author: Generated for Task 3 - Noisy velocity prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class SimpleGaussianNoise:
    """
    Simple i.i.d. Gaussian noise for baseline comparisons.
    
    This applies simple additive white Gaussian noise to all inputs
    with configurable standard deviations for SDF and velocity fields.
    
    Use this for baseline experiments before moving to the more complex
    OpticalFlowNoise model.
    
    Args:
        sdf_noise_std: Standard deviation of Gaussian noise for SDF
        vel_noise_std: Standard deviation of Gaussian noise for velocity
        enabled: Whether to apply noise
    """
    
    def __init__(
        self,
        sdf_noise_std: float = 0.1,
        vel_noise_std: float = 0.05,
        enabled: bool = True
    ):
        self.enabled = enabled
        self.sdf_noise_std = sdf_noise_std
        self.vel_noise_std = vel_noise_std
    
    @classmethod
    def from_config(cls, noise_cfg: Optional[Dict[str, Any]]) -> 'SimpleGaussianNoise':
        """
        Create noise model from configuration dictionary.
        
        Args:
            noise_cfg: Dictionary with noise parameters, or None for defaults
            
        Returns:
            Configured SimpleGaussianNoise instance
        """
        if noise_cfg is None:
            return cls(enabled=False)
        
        return cls(
            sdf_noise_std=noise_cfg.get('sdf_noise_std', 0.1),
            vel_noise_std=noise_cfg.get('vel_noise_std', 0.05),
            enabled=noise_cfg.get('enabled', True)
        )
    
    def __call__(
        self, 
        sdf: torch.Tensor, 
        velx_interface: torch.Tensor, 
        vely_interface: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply simple Gaussian noise to SDF and interface velocity.
        
        Args:
            sdf: Signed distance function [B, 1, H, W] or [1, H, W] or [H, W]
            velx_interface: X-component of interface velocity (same shapes)
            vely_interface: Y-component of interface velocity (same shapes)
            
        Returns:
            Tuple of (noisy_sdf, noisy_velx, noisy_vely)
        """
        if not self.enabled:
            return sdf, velx_interface, vely_interface
        
        device = sdf.device
        dtype = sdf.dtype
        
        # Add simple Gaussian noise
        noisy_sdf = sdf + torch.randn_like(sdf) * self.sdf_noise_std
        noisy_velx = velx_interface + torch.randn_like(velx_interface) * self.vel_noise_std
        noisy_vely = vely_interface + torch.randn_like(vely_interface) * self.vel_noise_std
        
        return noisy_sdf, noisy_velx, noisy_vely
    
    def __repr__(self) -> str:
        return (
            f"SimpleGaussianNoise(\n"
            f"  enabled={self.enabled},\n"
            f"  sdf_noise_std={self.sdf_noise_std},\n"
            f"  vel_noise_std={self.vel_noise_std}\n"
            f")"
        )


class SpatiallyCorrelatedNoise:
    """
    Generate spatially correlated noise using Gaussian filtering.
    
    This produces smooth random fields where neighboring pixels have
    correlated noise values, mimicking the spatial coherence in
    optical flow estimation errors.
    
    Args:
        correlation_length: Standard deviation of Gaussian blur kernel.
                           Higher values = more spatial correlation.
        noise_std: Standard deviation of the base white noise.
    """
    
    def __init__(self, correlation_length: float = 3.0, noise_std: float = 1.0):
        self.correlation_length = correlation_length
        self.noise_std = noise_std
        self._kernel = None
        self._kernel_size = None
    
    def _create_gaussian_kernel(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Create a 2D Gaussian kernel for blurring."""
        # Kernel size should be at least 6*sigma to capture most of the Gaussian
        kernel_size = int(6 * self.correlation_length) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create 1D Gaussian
        x = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
        gauss_1d = torch.exp(-x ** 2 / (2 * self.correlation_length ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        # Create 2D Gaussian via outer product
        kernel = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        kernel = kernel / kernel.sum()
        
        # Reshape for conv2d: (out_channels, in_channels, H, W)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        
        return kernel, kernel_size
    
    def __call__(self, shape: Tuple[int, ...], device: torch.device, 
                 dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Generate spatially correlated noise.
        
        Args:
            shape: Shape of noise tensor (B, C, H, W) or (C, H, W) or (H, W)
            device: Device to create noise on
            dtype: Data type
            
        Returns:
            Spatially correlated noise tensor of given shape
        """
        # Generate white noise
        white_noise = torch.randn(shape, device=device, dtype=dtype)
        
        if self.correlation_length <= 0:
            return white_noise * self.noise_std
        
        # Get or create kernel
        kernel, kernel_size = self._create_gaussian_kernel(device, dtype)
        
        # Handle different input shapes
        original_shape = shape
        if len(shape) == 2:
            white_noise = white_noise.unsqueeze(0).unsqueeze(0)  # Add B, C dims
        elif len(shape) == 3:
            white_noise = white_noise.unsqueeze(0)  # Add B dim
        
        B, C, H, W = white_noise.shape
        
        # Apply Gaussian blur channel-wise
        padding = kernel_size // 2
        blurred = []
        for c in range(C):
            channel = white_noise[:, c:c+1, :, :]
            blurred_channel = F.conv2d(channel, kernel, padding=padding)
            blurred.append(blurred_channel)
        
        blurred = torch.cat(blurred, dim=1)
        
        # Normalize to maintain unit variance, then scale by noise_std
        # Blurring reduces variance, so we compensate
        normalization_factor = math.sqrt(2 * math.pi) * self.correlation_length
        blurred = blurred * normalization_factor * self.noise_std
        
        # Restore original shape
        if len(original_shape) == 2:
            blurred = blurred.squeeze(0).squeeze(0)
        elif len(original_shape) == 3:
            blurred = blurred.squeeze(0)
        
        return blurred


class GradientAwareNoise:
    """
    Generate noise with higher magnitude at edges/gradients.
    
    This models the fact that optical flow has higher uncertainty at
    object boundaries where the SDF changes rapidly. Noise magnitude
    scales with the local gradient magnitude.
    
    Args:
        base_noise_std: Baseline noise standard deviation
        gradient_scale: How much to scale noise by gradient magnitude
        correlation_length: Spatial correlation of the noise (0 = uncorrelated)
    """
    
    def __init__(self, base_noise_std: float = 0.1, gradient_scale: float = 0.5,
                 correlation_length: float = 2.0):
        self.base_noise_std = base_noise_std
        self.gradient_scale = gradient_scale
        self.spatially_correlated = SpatiallyCorrelatedNoise(
            correlation_length=correlation_length, 
            noise_std=1.0
        )
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Generate gradient-aware noise for the input data.
        
        Args:
            data: Input tensor (B, C, H, W) or (C, H, W) or (H, W)
            
        Returns:
            Noise tensor with higher magnitude at edges
        """
        device = data.device
        dtype = data.dtype
        original_shape = data.shape
        
        # Ensure 4D tensor for gradient computation
        if len(original_shape) == 2:
            data_4d = data.unsqueeze(0).unsqueeze(0)
        elif len(original_shape) == 3:
            data_4d = data.unsqueeze(0)
        else:
            data_4d = data
        
        B, C, H, W = data_4d.shape
        
        # Compute gradient magnitude using Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               device=device, dtype=dtype).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               device=device, dtype=dtype).view(1, 1, 3, 3)
        
        # Compute gradients for each channel
        grad_magnitudes = []
        for c in range(C):
            channel = data_4d[:, c:c+1, :, :]
            grad_x = F.conv2d(channel, sobel_x, padding=1)
            grad_y = F.conv2d(channel, sobel_y, padding=1)
            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
            grad_magnitudes.append(grad_mag)
        
        grad_magnitude = torch.cat(grad_magnitudes, dim=1)
        
        # Normalize gradient magnitude to [0, 1] per sample
        grad_min = grad_magnitude.amin(dim=(-2, -1), keepdim=True)
        grad_max = grad_magnitude.amax(dim=(-2, -1), keepdim=True)
        grad_normalized = (grad_magnitude - grad_min) / (grad_max - grad_min + 1e-8)
        
        # Noise variance scales with gradient: var = base^2 + (gradient_scale * grad)^2
        noise_std_map = torch.sqrt(
            self.base_noise_std ** 2 + (self.gradient_scale * grad_normalized) ** 2
        )
        
        # Generate spatially correlated base noise
        base_noise = self.spatially_correlated(data_4d.shape, device, dtype)
        
        # Scale noise by the spatially varying standard deviation
        noise = base_noise * noise_std_map
        
        # Restore original shape
        if len(original_shape) == 2:
            noise = noise.squeeze(0).squeeze(0)
        elif len(original_shape) == 3:
            noise = noise.squeeze(0)
        
        return noise


class VelocityScaledNoise:
    """
    Generate noise that scales with velocity magnitude.
    
    Optical flow estimation errors typically scale with the flow magnitude
    (faster motion = higher uncertainty). This noise model reflects that.
    
    Args:
        base_noise_std: Baseline noise (even for zero velocity)
        velocity_scale: How much noise scales with |velocity|
        correlation_length: Spatial correlation of noise
    """
    
    def __init__(self, base_noise_std: float = 0.05, velocity_scale: float = 0.1,
                 correlation_length: float = 2.0):
        self.base_noise_std = base_noise_std
        self.velocity_scale = velocity_scale
        self.spatially_correlated = SpatiallyCorrelatedNoise(
            correlation_length=correlation_length,
            noise_std=1.0
        )
    
    def __call__(self, velx: torch.Tensor, vely: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate velocity-scaled noise for velocity components.
        
        Args:
            velx: X-component of velocity
            vely: Y-component of velocity
            
        Returns:
            Tuple of (noise_x, noise_y) tensors
        """
        device = velx.device
        dtype = velx.dtype
        
        # Compute velocity magnitude
        vel_magnitude = torch.sqrt(velx ** 2 + vely ** 2 + 1e-8)
        
        # Noise standard deviation scales with velocity magnitude
        noise_std = self.base_noise_std + self.velocity_scale * vel_magnitude
        
        # Generate spatially correlated noise for each component
        noise_x = self.spatially_correlated(velx.shape, device, dtype) * noise_std
        noise_y = self.spatially_correlated(vely.shape, device, dtype) * noise_std
        
        return noise_x, noise_y


class OpticalFlowNoise:
    """
    Combined noise model simulating optical flow estimation uncertainty.
    
    This is the main interface for adding realistic optical flow noise to
    SDF and interface velocity fields. It combines:
    1. Spatially correlated noise (smooth random fields)
    2. Gradient-aware noise for SDF (higher at bubble boundaries)
    3. Velocity-scaled noise for interface velocity
    
    Usage:
        noise_model = OpticalFlowNoise(cfg)
        noisy_sdf, noisy_velx, noisy_vely = noise_model(sdf, velx, vely)
    
    Args:
        sdf_noise_std: Base noise level for SDF
        sdf_gradient_scale: Additional noise at SDF gradients (bubble edges)
        vel_base_noise_std: Base noise for velocity
        vel_scale_factor: How much velocity noise scales with |velocity|
        correlation_length: Spatial correlation length (in pixels)
        enabled: Whether to apply noise (useful for toggling during eval)
    """
    
    def __init__(
        self,
        sdf_noise_std: float = 0.1,
        sdf_gradient_scale: float = 0.3,
        vel_base_noise_std: float = 0.05,
        vel_scale_factor: float = 0.15,
        correlation_length: float = 3.0,
        enabled: bool = True
    ):
        self.enabled = enabled
        self.sdf_noise_std = sdf_noise_std
        self.sdf_gradient_scale = sdf_gradient_scale
        self.vel_base_noise_std = vel_base_noise_std
        self.vel_scale_factor = vel_scale_factor
        self.correlation_length = correlation_length
        
        # Initialize noise generators
        self.sdf_noise = GradientAwareNoise(
            base_noise_std=sdf_noise_std,
            gradient_scale=sdf_gradient_scale,
            correlation_length=correlation_length
        )
        
        self.velocity_noise = VelocityScaledNoise(
            base_noise_std=vel_base_noise_std,
            velocity_scale=vel_scale_factor,
            correlation_length=correlation_length
        )
    
    @classmethod
    def from_config(cls, noise_cfg: Optional[Dict[str, Any]]) -> 'OpticalFlowNoise':
        """
        Create noise model from configuration dictionary.
        
        Args:
            noise_cfg: Dictionary with noise parameters, or None for defaults
            
        Returns:
            Configured OpticalFlowNoise instance
        """
        if noise_cfg is None:
            return cls(enabled=False)
        
        return cls(
            sdf_noise_std=noise_cfg.get('sdf_noise_std', 0.1),
            sdf_gradient_scale=noise_cfg.get('sdf_gradient_scale', 0.3),
            vel_base_noise_std=noise_cfg.get('vel_base_noise_std', 0.05),
            vel_scale_factor=noise_cfg.get('vel_scale_factor', 0.15),
            correlation_length=noise_cfg.get('correlation_length', 3.0),
            enabled=noise_cfg.get('enabled', True)
        )
    
    def __call__(
        self, 
        sdf: torch.Tensor, 
        velx_interface: torch.Tensor, 
        vely_interface: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply optical flow noise to SDF and interface velocity.
        
        Args:
            sdf: Signed distance function [B, 1, H, W] or [1, H, W] or [H, W]
            velx_interface: X-component of interface velocity (same shapes)
            vely_interface: Y-component of interface velocity (same shapes)
            
        Returns:
            Tuple of (noisy_sdf, noisy_velx, noisy_vely)
        """
        if not self.enabled:
            return sdf, velx_interface, vely_interface
        
        # Add gradient-aware noise to SDF (higher noise at bubble boundaries)
        sdf_noise = self.sdf_noise(sdf)
        noisy_sdf = sdf + sdf_noise
        
        # Add velocity-scaled noise to interface velocity
        vel_noise_x, vel_noise_y = self.velocity_noise(velx_interface, vely_interface)
        noisy_velx = velx_interface + vel_noise_x
        noisy_vely = vely_interface + vel_noise_y
        
        return noisy_sdf, noisy_velx, noisy_vely
    
    def __repr__(self) -> str:
        return (
            f"OpticalFlowNoise(\n"
            f"  enabled={self.enabled},\n"
            f"  sdf_noise_std={self.sdf_noise_std},\n"
            f"  sdf_gradient_scale={self.sdf_gradient_scale},\n"
            f"  vel_base_noise_std={self.vel_base_noise_std},\n"
            f"  vel_scale_factor={self.vel_scale_factor},\n"
            f"  correlation_length={self.correlation_length}\n"
            f")"
        )


def create_noise_model(noise_cfg: Optional[Dict[str, Any]]):
    """
    Factory function to create appropriate noise model based on configuration.
    
    Args:
        noise_cfg: Dictionary with noise parameters containing:
            - noise_type: "gaussian" for simple noise, "optical_flow" for complex noise
            - enabled: Whether to apply noise
            - Other parameters specific to the noise type
            
    Returns:
        Noise model instance (SimpleGaussianNoise or OpticalFlowNoise)
        
    Examples:
        # Simple Gaussian noise
        noise_cfg = {
            "enabled": True,
            "noise_type": "gaussian",
            "sdf_noise_std": 0.1,
            "vel_noise_std": 0.05
        }
        
        # Complex optical flow noise
        noise_cfg = {
            "enabled": True,
            "noise_type": "optical_flow",
            "sdf_noise_std": 0.1,
            "sdf_gradient_scale": 0.3,
            "vel_base_noise_std": 0.05,
            "vel_scale_factor": 0.15,
            "correlation_length": 3.0
        }
    """
    if noise_cfg is None:
        return OpticalFlowNoise(enabled=False)
    
    if not noise_cfg.get('enabled', True):
        return OpticalFlowNoise(enabled=False)
    
    noise_type = noise_cfg.get('noise_type', 'optical_flow').lower()
    
    if noise_type == 'gaussian' or noise_type == 'simple':
        return SimpleGaussianNoise.from_config(noise_cfg)
    elif noise_type == 'optical_flow' or noise_type == 'complex':
        return OpticalFlowNoise.from_config(noise_cfg)
    else:
        raise ValueError(
            f"Unknown noise_type: '{noise_type}'. "
            f"Supported types: 'gaussian', 'simple', 'optical_flow', 'complex'"
        )
