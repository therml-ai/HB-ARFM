"""
Autoregressive Conditional Flow Matching for BubbleFlow Prediction.

This module implements an Autoregressive Conditional Flow Matching model
that conditions on the previous timestep output to enforce temporal consistency.

Key Idea:
- Standard flow matching with autoregressive conditioning
- Model conditions on: [current_state_noisy, conditioning_t, output_{t-1}]
- During training: uses teacher forcing (ground truth previous state)
- During inference: uses model's own predictions (autoregressive rollout)

Training Modes:
1. Teacher Forcing (scheduled_sampling.enabled=False):
   - Previous state always comes from ground truth
   - Efficient but may cause exposure bias

2. Scheduled Sampling (scheduled_sampling.enabled=True):
   - Gradually transitions from teacher forcing to model predictions
   - Early: 100% ground truth (stability)
   - Mid: mixture of ground truth and predictions
   - Late: 100% predictions (like inference)
   - Reduces exposure bias

Autoregressive Formulation:
    Input:  [conditioning_t, output_{t-1}]  # [B, 6, H, W] for Task 2
    Target: output_t                         # [B, 3, H, W]

Where:
    - conditioning_t = [SDF_t, velx_interface_t, vely_interface_t]
    - output_{t-1} = [velx_{t-1}, vely_{t-1}, temp_{t-1}] (teacher forcing or predicted)
    - output_t = [velx_t, vely_t, temp_t]

This encourages temporal consistency by construction, as the model must
learn to produce outputs that are consistent with its previous predictions.

Classes:
    ConditionalFlowMatchingAR: OT-CFM with autoregressive conditioning
    ConditionalFlowMatchingARLightning: PyTorch Lightning wrapper

References:
    - "Flow Matching for Generative Modeling" (Lipman et al., 2023)
    - "Scheduled Sampling for Sequence Prediction with RNNs" (Bengio et al., 2015)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from omegaconf import DictConfig
from typing import Optional, Tuple
import numpy as np

from bubblefusion.models.flow_matching import (
    FlowMatchingUNet,
    FlowMatchingSampler,
    TimeEmbedding,
    ResidualBlock,
    AttentionBlock,
    Upsample,
    Downsample,
)


class SpectralLoss(nn.Module):
    """
    Spectral loss to preserve high-frequency details.
    
    Computes loss in frequency domain to prevent blur/smoothing.
    Weights higher frequencies more heavily to encourage sharp predictions.
    
    This is critical for flow matching AR models which tend to produce
    blurry outputs due to:
    1. MSE regression-to-mean behavior
    2. Error accumulation in AR rollout
    """
    
    def __init__(
        self, 
        weight: float = 0.1,
        high_freq_weight: float = 2.0,
        freq_threshold: float = 0.3,
    ):
        """
        Args:
            weight: Overall weight of spectral loss
            high_freq_weight: Extra weight for high frequencies
            freq_threshold: Threshold (0-1) above which frequencies are "high"
        """
        super().__init__()
        self.weight = weight
        self.high_freq_weight = high_freq_weight
        self.freq_threshold = freq_threshold
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spectral loss between prediction and target.
        
        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]
            
        Returns:
            Weighted spectral loss scalar
        """
        # Compute 2D FFT
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        
        # Compute magnitude spectra
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # Compute phase difference (wrapped)
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        phase_diff = torch.abs(pred_phase - target_phase)
        # Wrap to [-pi, pi]
        phase_diff = torch.minimum(phase_diff, 2*np.pi - phase_diff)
        
        # Create frequency weight mask (higher weight for high frequencies)
        B, C, H, W = pred.shape
        freq_y = torch.fft.fftfreq(H, device=pred.device).view(-1, 1).abs()
        freq_x = torch.fft.fftfreq(W, device=pred.device).view(1, -1).abs()
        freq_magnitude = torch.sqrt(freq_y**2 + freq_x**2)
        
        # Normalize to [0, 1] range
        freq_magnitude = freq_magnitude / freq_magnitude.max()
        
        # Weight mask: 1.0 for low freq, high_freq_weight for high freq
        weight_mask = torch.where(
            freq_magnitude > self.freq_threshold,
            torch.full_like(freq_magnitude, self.high_freq_weight),
            torch.ones_like(freq_magnitude)
        )
        weight_mask = weight_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Weighted magnitude loss
        mag_loss = ((pred_mag - target_mag).abs() * weight_mask).mean()
        
        # Phase loss (less weighted, mostly for coherence)
        phase_loss = (phase_diff * weight_mask).mean() * 0.1
        
        return self.weight * (mag_loss + phase_loss)


class GradientLoss(nn.Module):
    """
    Gradient loss to preserve edges and sharp transitions.
    
    Computes spatial gradients and penalizes differences,
    encouraging the model to maintain sharp boundaries.
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
        
        # Sobel kernels for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient loss.
        
        Args:
            pred: [B, C, H, W]
            target: [B, C, H, W]
            
        Returns:
            Gradient loss scalar
        """
        B, C, H, W = pred.shape
        
        total_loss = 0.0
        for c in range(C):
            pred_c = pred[:, c:c+1, :, :]
            target_c = target[:, c:c+1, :, :]
            
            # Compute gradients
            pred_gx = F.conv2d(pred_c, self.sobel_x, padding=1)
            pred_gy = F.conv2d(pred_c, self.sobel_y, padding=1)
            target_gx = F.conv2d(target_c, self.sobel_x, padding=1)
            target_gy = F.conv2d(target_c, self.sobel_y, padding=1)
            
            # L1 loss on gradients
            loss_x = F.l1_loss(pred_gx, target_gx)
            loss_y = F.l1_loss(pred_gy, target_gy)
            
            total_loss = total_loss + loss_x + loss_y
        
        return self.weight * total_loss / C


# =============================================================================
# PHYSICS-INFORMED LOSSES
# =============================================================================
# These losses enforce physical constraints learned from the governing equations:
# 1. Divergence-Free Loss: Mass conservation (∇·V = 0 for incompressible flow)
# 2. Vorticity Consistency Loss: Rotational flow patterns (ω = ∂v/∂x - ∂u/∂y)
# 3. Advection Consistency Loss: Velocity-temperature coupling (∂T/∂t + u·∇T ≈ 0)
# =============================================================================


class DivergenceLoss(nn.Module):
    """
    Divergence-Free Loss for Mass Conservation.
    
    For incompressible flow, ∇·V = ∂u/∂x + ∂v/∂y should be zero.
    This loss penalizes non-zero divergence in the predicted velocity field,
    excluding the interface region where phase change causes mass flux.
    
    Physics Background:
    - Continuity equation: ∂ρ/∂t + ∇·(ρV) = 0
    - For incompressible flow (ρ = const): ∇·V = 0
    - At interface: Mass flux from phase change violates this locally
    
    Interface Detection (matching physics_metrics_task2.py):
    - For VELOCITY metrics: Use MASSFLUX-based detection
      Interface cells = where |velx_interface| OR |vely_interface| > threshold
      This is more physically meaningful because velocity has discontinuity
      where there's actual mass transfer, not just at geometric interface.
    
    Grid Notes (BubbleML):
    - Domain: 16 x 16 (non-dimensional)
    - Full resolution: 512 x 512
    - dx = dy = domain_size / resolution = 16/512 = 1/32 (at full res)
    - With downsample_factor: dx = downsample_factor * (16/512) = downsample_factor/32
    """
    
    def __init__(
        self, 
        weight: float = 0.1,
        downsample_factor: int = 1,
        massflux_threshold: float = 1e-2,
        exclude_interface: bool = True,
    ):
        """
        Args:
            weight: Loss weight (suggest 0.05-0.2)
            downsample_factor: Spatial downsampling factor (affects grid spacing)
            massflux_threshold: Threshold for interface velocity magnitude (default 1e-2)
                               Cells with |velx_interface| or |vely_interface| > threshold
                               are considered interface cells with active phase change.
            exclude_interface: If True, exclude interface from loss computation
        """
        super().__init__()
        self.weight = weight
        self.dx = downsample_factor / 32.0  # Grid spacing in non-dimensional units
        self.massflux_threshold = massflux_threshold
        self.exclude_interface = exclude_interface
        
    def forward(
        self, 
        pred_velx: torch.Tensor,  # [B, H, W] or [B, 1, H, W]
        pred_vely: torch.Tensor,  # [B, H, W] or [B, 1, H, W]
        velx_interface: torch.Tensor = None,  # [B, H, W] for massflux-based interface detection
        vely_interface: torch.Tensor = None,  # [B, H, W] for massflux-based interface detection
    ) -> torch.Tensor:
        """
        Compute divergence loss.
        
        Args:
            pred_velx: Predicted x-velocity [B, H, W] or [B, 1, H, W]
            pred_vely: Predicted y-velocity [B, H, W] or [B, 1, H, W]
            velx_interface: Interface x-velocity for massflux-based masking
            vely_interface: Interface y-velocity for massflux-based masking
            
        Returns:
            Weighted divergence loss scalar
        """
        # Handle channel dimension
        if pred_velx.dim() == 4:
            pred_velx = pred_velx.squeeze(1)
            pred_vely = pred_vely.squeeze(1)
        if velx_interface is not None and velx_interface.dim() == 4:
            velx_interface = velx_interface.squeeze(1)
            vely_interface = vely_interface.squeeze(1)
        
        # Compute velocity divergence using central differences
        # dudx = (u[i+1] - u[i-1]) / (2*dx)
        # Note: gradient on axis=-1 is x-direction (W), axis=-2 is y-direction (H)
        dudx = (pred_velx[:, :, 2:] - pred_velx[:, :, :-2]) / (2 * self.dx)
        dvdy = (pred_vely[:, 2:, :] - pred_vely[:, :-2, :]) / (2 * self.dx)
        
        # Align spatial dimensions (both are now [B, H-2, W-2])
        dudx = dudx[:, 1:-1, :]  # [B, H-2, W-2]
        dvdy = dvdy[:, :, 1:-1]  # [B, H-2, W-2]
        
        divergence = dudx + dvdy  # [B, H-2, W-2]
        
        # Exclude interface region using MASSFLUX-based detection
        # (matching physics_metrics_task2.py for velocity metrics)
        if self.exclude_interface and velx_interface is not None and vely_interface is not None:
            # Crop interface velocities to match divergence dimensions
            velx_int_crop = velx_interface[:, 1:-1, 1:-1]  # [B, H-2, W-2]
            vely_int_crop = vely_interface[:, 1:-1, 1:-1]  # [B, H-2, W-2]
            
            # Interface mask: cells with non-zero interface velocity (mass flux)
            interface_mask = ((velx_int_crop.abs() >= self.massflux_threshold) | 
                             (vely_int_crop.abs() >= self.massflux_threshold))
            
            # Bulk mask: 1.0 for bulk (no mass flux), 0.0 for interface
            bulk_mask = (~interface_mask).float()
            
            # Compute masked loss
            div_squared = divergence ** 2
            masked_div = div_squared * bulk_mask
            
            # Normalize by number of bulk points
            num_bulk = bulk_mask.sum() + 1e-8
            div_loss = masked_div.sum() / num_bulk
        else:
            div_loss = (divergence ** 2).mean()
        
        return self.weight * div_loss


class VorticityLoss(nn.Module):
    """
    Vorticity Consistency Loss.
    
    Matches vorticity (ω = ∂v/∂x - ∂u/∂y) between prediction and target.
    Vorticity captures rotational flow structures like eddies and recirculation.
    
    Physics Background:
    - Vorticity ω = curl(V) = ∂v/∂x - ∂u/∂y (in 2D)
    - Positive ω: counter-clockwise rotation
    - Negative ω: clockwise rotation
    - Important for capturing wake structures behind bubbles
    
    Interface Detection (matching physics_metrics_task2.py):
    - For VELOCITY metrics: Use MASSFLUX-based detection
      Interface cells = where |velx_interface| OR |vely_interface| > threshold
    
    Why This Matters:
    - Your metrics show 10x error in vorticity in bulk liquid
    - Matching vorticity ensures correct rotational dynamics
    - Helps preserve flow structures around rising bubbles
    """
    
    def __init__(
        self, 
        weight: float = 0.1,
        downsample_factor: int = 1,
        massflux_threshold: float = 1e-2,
        exclude_interface: bool = True,
    ):
        """
        Args:
            weight: Loss weight (suggest 0.05-0.1)
            downsample_factor: Spatial downsampling factor
            massflux_threshold: Threshold for interface velocity magnitude (default 1e-2)
            exclude_interface: If True, exclude interface from loss computation
        """
        super().__init__()
        self.weight = weight
        self.dx = downsample_factor / 32.0
        self.massflux_threshold = massflux_threshold
        self.exclude_interface = exclude_interface
        
    def _compute_vorticity(
        self, 
        velx: torch.Tensor, 
        vely: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute vorticity field ω = ∂v/∂x - ∂u/∂y.
        
        Args:
            velx: x-velocity [B, H, W]
            vely: y-velocity [B, H, W]
            
        Returns:
            vorticity: [B, H-2, W-2]
        """
        # dvdx = ∂v/∂x using central differences
        dvdx = (vely[:, :, 2:] - vely[:, :, :-2]) / (2 * self.dx)
        # dudy = ∂u/∂y using central differences
        dudy = (velx[:, 2:, :] - velx[:, :-2, :]) / (2 * self.dx)
        
        # Align dimensions
        dvdx = dvdx[:, 1:-1, :]  # [B, H-2, W-2]
        dudy = dudy[:, :, 1:-1]  # [B, H-2, W-2]
        
        return dvdx - dudy
    
    def forward(
        self, 
        pred_velx: torch.Tensor,
        pred_vely: torch.Tensor,
        target_velx: torch.Tensor,
        target_vely: torch.Tensor,
        velx_interface: torch.Tensor = None,
        vely_interface: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute vorticity consistency loss.
        
        Args:
            pred_velx: Predicted x-velocity [B, H, W] or [B, 1, H, W]
            pred_vely: Predicted y-velocity [B, H, W] or [B, 1, H, W]
            target_velx: Target x-velocity
            target_vely: Target y-velocity
            velx_interface: Interface x-velocity for massflux-based masking
            vely_interface: Interface y-velocity for massflux-based masking
            
        Returns:
            Weighted vorticity loss scalar
        """
        # Handle channel dimension
        if pred_velx.dim() == 4:
            pred_velx = pred_velx.squeeze(1)
            pred_vely = pred_vely.squeeze(1)
            target_velx = target_velx.squeeze(1)
            target_vely = target_vely.squeeze(1)
        if velx_interface is not None and velx_interface.dim() == 4:
            velx_interface = velx_interface.squeeze(1)
            vely_interface = vely_interface.squeeze(1)
        
        # Compute vorticity
        pred_vort = self._compute_vorticity(pred_velx, pred_vely)
        target_vort = self._compute_vorticity(target_velx, target_vely)
        
        # Compute error
        vort_error = (pred_vort - target_vort) ** 2
        
        # Exclude interface region using MASSFLUX-based detection
        # (matching physics_metrics_task2.py for velocity metrics)
        if self.exclude_interface and velx_interface is not None and vely_interface is not None:
            # Crop interface velocities to match vorticity dimensions
            velx_int_crop = velx_interface[:, 1:-1, 1:-1]  # [B, H-2, W-2]
            vely_int_crop = vely_interface[:, 1:-1, 1:-1]  # [B, H-2, W-2]
            
            # Interface mask: cells with non-zero interface velocity (mass flux)
            interface_mask = ((velx_int_crop.abs() >= self.massflux_threshold) | 
                             (vely_int_crop.abs() >= self.massflux_threshold))
            
            # Bulk mask: 1.0 for bulk (no mass flux), 0.0 for interface
            bulk_mask = (~interface_mask).float()
            
            masked_error = vort_error * bulk_mask
            num_bulk = bulk_mask.sum() + 1e-8
            vort_loss = masked_error.sum() / num_bulk
        else:
            vort_loss = vort_error.mean()
        
        return self.weight * vort_loss


class AdvectionConsistencyLoss(nn.Module):
    """
    Advection Consistency Loss - Temperature-Velocity Coupling.
    
    This is the key physics loss suggested by your advisor. It enforces that
    temperature evolves consistently with advection by the predicted velocity.
    
    Physics Background (Energy Equation LHS):
        ∂T/∂t + u·∇T = RHS (diffusion + sources)
    
    Key Insight (from your advisor):
    - Focus on LHS (advection), ignore RHS (stiff near interface)
    - Check if temperature moves in the direction velocity says it should
    - This is a geometric constraint, not solving the full PDE
    
    Formulation:
        T_advected = T_{t-1} - Δt * (u_{t-1} * ∂T_{t-1}/∂x + v_{t-1} * ∂T_{t-1}/∂y)
        
        Loss = ||T_t^pred - T_advected||_2
    
    This checks: Does the temperature increment (T_t - T_{t-1}) align with
    the velocity-induced transport direction?
    
    Notes:
    - Only apply in bulk liquid (away from interface where RHS dominates)
    - Uses previous state velocity and temperature (available during training)
    - Δt is the simulation time step between frames
    """
    
    def __init__(
        self, 
        weight: float = 0.1,
        downsample_factor: int = 1,
        dt: float = 0.001,  # Time step between frames (non-dimensional)
        interface_threshold: float = 0.1,
        exclude_interface: bool = True,
        exclude_near_wall: bool = True,
        wall_rows: int = 4,  # Number of rows near wall to exclude
    ):
        """
        Args:
            weight: Loss weight (suggest 0.05-0.1)
            downsample_factor: Spatial downsampling factor
            dt: Time step between consecutive frames (non-dimensional, ~0.001 typical)
            interface_threshold: SDF threshold for interface region
            exclude_interface: If True, exclude interface from loss
            exclude_near_wall: If True, exclude near-wall region (heat source)
            wall_rows: Number of rows near bottom wall to exclude
        """
        super().__init__()
        self.weight = weight
        self.dx = downsample_factor / 32.0
        self.dt = dt
        self.interface_threshold = interface_threshold
        self.exclude_interface = exclude_interface
        self.exclude_near_wall = exclude_near_wall
        self.wall_rows = wall_rows
        
    def forward(
        self, 
        pred_temp: torch.Tensor,      # T_t^pred [B, H, W] or [B, 1, H, W]
        prev_temp: torch.Tensor,      # T_{t-1} [B, H, W] or [B, 1, H, W]
        prev_velx: torch.Tensor,      # u_{t-1} [B, H, W] or [B, 1, H, W]
        prev_vely: torch.Tensor,      # v_{t-1} [B, H, W] or [B, 1, H, W]
        sdf: torch.Tensor = None,     # For interface masking
    ) -> torch.Tensor:
        """
        Compute advection consistency loss.
        
        Checks if T_t^pred is consistent with advecting T_{t-1} using velocity.
        
        Args:
            pred_temp: Predicted temperature at time t
            prev_temp: Temperature at time t-1 (from prev_output)
            prev_velx: x-velocity at time t-1 (from prev_output)
            prev_vely: y-velocity at time t-1 (from prev_output)
            sdf: Signed distance function at time t for interface masking
            
        Returns:
            Weighted advection consistency loss scalar
        """
        # Handle channel dimension
        if pred_temp.dim() == 4:
            pred_temp = pred_temp.squeeze(1)
            prev_temp = prev_temp.squeeze(1)
            prev_velx = prev_velx.squeeze(1)
            prev_vely = prev_vely.squeeze(1)
        if sdf is not None and sdf.dim() == 4:
            sdf = sdf.squeeze(1)
        
        # Compute temperature gradients using central differences
        # dTdx = ∂T/∂x
        dTdx = (prev_temp[:, :, 2:] - prev_temp[:, :, :-2]) / (2 * self.dx)
        # dTdy = ∂T/∂y
        dTdy = (prev_temp[:, 2:, :] - prev_temp[:, :-2, :]) / (2 * self.dx)
        
        # Crop velocity to match gradient dimensions
        u_crop = prev_velx[:, 1:-1, 1:-1]  # [B, H-2, W-2]
        v_crop = prev_vely[:, 1:-1, 1:-1]  # [B, H-2, W-2]
        
        # Align gradient dimensions
        dTdx = dTdx[:, 1:-1, :]  # [B, H-2, W-2]
        dTdy = dTdy[:, :, 1:-1]  # [B, H-2, W-2]
        
        # Compute advection term: u·∇T = u * ∂T/∂x + v * ∂T/∂y
        advection_term = u_crop * dTdx + v_crop * dTdy
        
        # Advected temperature: T_advected = T_{t-1} - dt * u·∇T
        # (Forward Euler discretization of advection equation)
        prev_temp_crop = prev_temp[:, 1:-1, 1:-1]
        temp_advected = prev_temp_crop - self.dt * advection_term
        
        # Crop predicted temperature to match
        pred_temp_crop = pred_temp[:, 1:-1, 1:-1]
        
        # Compute error
        adv_error = (pred_temp_crop - temp_advected) ** 2
        
        # Create mask for valid regions (bulk liquid, away from wall)
        B, H, W = adv_error.shape
        mask = torch.ones_like(adv_error)
        
        # Exclude interface region
        if self.exclude_interface and sdf is not None:
            sdf_crop = sdf[:, 1:-1, 1:-1]
            interface_mask = (sdf_crop.abs() < self.interface_threshold).float()
            mask = mask * (1.0 - interface_mask)
        
        # Exclude near-wall region (where heat source dominates, not advection)
        if self.exclude_near_wall:
            # Wall is at y=0 (first rows in array)
            wall_mask = torch.zeros_like(mask)
            wall_mask[:, :min(self.wall_rows, H), :] = 1.0
            mask = mask * (1.0 - wall_mask)
        
        # Compute masked loss
        masked_error = adv_error * mask
        num_valid = mask.sum() + 1e-8
        adv_loss = masked_error.sum() / num_valid
        
        return self.weight * adv_loss


class ConditionalFlowMatchingAR(nn.Module):
    """
    Autoregressive Optimal Transport Conditional Flow Matching model.
    
    This model extends standard flow matching with autoregressive conditioning:
    - Conditions on both current inputs AND previous timestep outputs
    - Uses teacher forcing during training (ground truth previous state)
    - Uses model predictions during inference (autoregressive rollout)
    
    Flow matching learns a continuous flow from noise to data:
        dx/dt = v_θ(x(t), t, condition_t, output_{t-1})
    
    Args:
        unet: The UNet velocity field predictor
    """
    
    def __init__(self, unet: FlowMatchingUNet):
        super().__init__()
        self.unet = unet
        
    def compute_conditional_flow(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute conditional flow path using optimal transport (linear interpolation).
        
        Args:
            x_0: Source samples (noise) [B, C, H, W]
            x_1: Target samples (clean output) [B, C, H, W]
            t: Time values in [0, 1] [B]
            
        Returns:
            x_t: Interpolated samples [B, C, H, W]
            velocity_target: Target velocity field [B, C, H, W]
        """
        t = t.view(-1, 1, 1, 1)
        x_t = (1 - t) * x_0 + t * x_1
        velocity_target = x_1 - x_0
        return x_t, velocity_target
    
    def forward(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        prev_output: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict velocity field given current state, conditioning, and previous output.
        
        Args:
            x_t: Current noisy state [B, C_out, H, W]
            condition: Current conditioning (SDF, interface vel) [B, C_cond, H, W]
            prev_output: Previous timestep output (teacher forcing) [B, C_out, H, W]
            t: Time values in [0, 1] [B]
            
        Returns:
            Predicted velocity field [B, C_out, H, W]
        """
        # Concatenate: [x_t, condition, prev_output]
        # For Task 2: [3, 3, 3] = 9 channels
        x_input = torch.cat([x_t, condition, prev_output], dim=1)
        return self.unet(x_input, t)
    
    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        prev_output: torch.Tensor,
        shape: Tuple[int, ...],
        device: torch.device,
        num_integration_steps: int = 50,
        solver: str = 'euler',
        guidance_scale: float = 1.0,
        stochastic_temperature: float = 0.0,
    ) -> torch.Tensor:
        """
        Generate samples using ODE/SDE integration with autoregressive conditioning.
        
        Args:
            condition: Current conditioning [B, C_cond, H, W]
            prev_output: Previous timestep output [B, C_out, H, W]
            shape: Output shape (B, C_out, H, W)
            device: Device for generation
            num_integration_steps: Number of integration steps
            solver: ODE solver - 'euler', 'heun', 'rk4', or 'midpoint'
            guidance_scale: Classifier-free guidance scale (1.0 = no guidance)
            stochastic_temperature: Temperature for stochastic sampling (0.0 = deterministic ODE,
                                   >0 = SDE with noise injection for diverse outputs).
                                   Recommended range: 0.01-0.1 for subtle diversity,
                                   0.1-0.5 for more exploration.
            
        Returns:
            Generated samples [B, C_out, H, W]
        """
        # Start from noise
        x = torch.randn(shape, device=device)
        
        dt = 1.0 / num_integration_steps
        
        # Compute noise scale for SDE (if stochastic_temperature > 0)
        # The noise decreases as we approach t=1 (the data distribution)
        use_stochastic = stochastic_temperature > 0.0
        
        # Integrate ODE/SDE from t=0 to t=1
        for step in range(num_integration_steps):
            t = step * dt
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.float32)
            
            # Compute stochastic noise term (decreases as t approaches 1)
            # noise_scale = temperature * sqrt(dt) * (1 - t) to reduce noise near data
            if use_stochastic and step < num_integration_steps - 1:
                noise_scale = stochastic_temperature * math.sqrt(dt) * (1.0 - t)
                stochastic_noise = noise_scale * torch.randn_like(x)
            else:
                stochastic_noise = 0.0
            
            if solver == 'euler':
                # Euler method (1st order)
                velocity = self._get_velocity_with_guidance(
                    x, condition, prev_output, t_tensor, guidance_scale
                )
                x = x + velocity * dt + stochastic_noise
                
            elif solver == 'heun':
                # Heun's method (2nd order, predictor-corrector)
                # Predictor (Euler)
                v1 = self._get_velocity_with_guidance(
                    x, condition, prev_output, t_tensor, guidance_scale
                )
                x_pred = x + v1 * dt
                
                # Corrector
                t_next = torch.full((shape[0],), min(t + dt, 1.0), device=device, dtype=torch.float32)
                v2 = self._get_velocity_with_guidance(
                    x_pred, condition, prev_output, t_next, guidance_scale
                )
                
                # Average velocities + stochastic term
                x = x + 0.5 * (v1 + v2) * dt + stochastic_noise
                
            elif solver == 'midpoint':
                # Midpoint method (2nd order)
                v1 = self._get_velocity_with_guidance(
                    x, condition, prev_output, t_tensor, guidance_scale
                )
                x_mid = x + v1 * (dt / 2)
                
                t_mid = torch.full((shape[0],), t + dt/2, device=device, dtype=torch.float32)
                v_mid = self._get_velocity_with_guidance(
                    x_mid, condition, prev_output, t_mid, guidance_scale
                )
                x = x + v_mid * dt + stochastic_noise
                
            elif solver == 'rk4':
                # Runge-Kutta 4th order
                t_half = torch.full((shape[0],), t + dt/2, device=device, dtype=torch.float32)
                t_next = torch.full((shape[0],), min(t + dt, 1.0), device=device, dtype=torch.float32)
                
                k1 = self._get_velocity_with_guidance(x, condition, prev_output, t_tensor, guidance_scale)
                k2 = self._get_velocity_with_guidance(x + k1 * dt/2, condition, prev_output, t_half, guidance_scale)
                k3 = self._get_velocity_with_guidance(x + k2 * dt/2, condition, prev_output, t_half, guidance_scale)
                k4 = self._get_velocity_with_guidance(x + k3 * dt, condition, prev_output, t_next, guidance_scale)
                
                x = x + (k1 + 2*k2 + 2*k3 + k4) * (dt / 6) + stochastic_noise
            else:
                raise ValueError(f"Unknown solver: {solver}. Use 'euler', 'heun', 'midpoint', or 'rk4'")
        
        return x
    
    def _get_velocity_with_guidance(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        prev_output: torch.Tensor,
        t: torch.Tensor,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Get velocity with optional classifier-free guidance.
        
        When guidance_scale > 1.0, amplifies the difference between
        conditioned and unconditioned predictions for sharper outputs.
        
        Args:
            x: Current state [B, C_out, H, W]
            condition: Conditioning [B, C_cond, H, W]
            prev_output: Previous output [B, C_out, H, W]
            t: Time values [B]
            guidance_scale: CFG scale (1.0 = no guidance, >1.0 = sharper)
            
        Returns:
            Guided velocity [B, C_out, H, W]
        """
        # Conditioned prediction
        velocity_cond = self(x, condition, prev_output, t)
        
        if guidance_scale == 1.0:
            return velocity_cond
        
        # Unconditioned prediction (zero out conditioning)
        # This simulates what the model would predict without guidance
        uncond_condition = torch.zeros_like(condition)
        uncond_prev = torch.zeros_like(prev_output)
        velocity_uncond = self(x, uncond_condition, uncond_prev, t)
        
        # Classifier-free guidance: amplify difference from unconditional
        # v_guided = v_uncond + scale * (v_cond - v_uncond)
        velocity_guided = velocity_uncond + guidance_scale * (velocity_cond - velocity_uncond)
        
        return velocity_guided
    
    @torch.no_grad()
    def create_initial_state(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        mode: str = 'zeros',
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Create an initial state for autoregressive inference when ground truth is not available.
        
        This is useful for pure inference scenarios where we don't have access to the
        ground truth previous state (e.g., deploying on new data with only SDF and 
        interface velocity inputs).
        
        Args:
            shape: Output shape (B, C_out, H, W)
            device: Device for the tensor
            mode: Initialization mode:
                - 'zeros': All zeros (neutral state for normalized data)
                - 'small_noise': Small random noise around zero (adds some variation)
                - 'from_conditioning': Attempt to derive from conditioning (experimental)
            conditioning: Optional conditioning tensor [B, C_cond, H, W] for 'from_conditioning' mode
            
        Returns:
            Initial state tensor [B, C_out, H, W]
            
        Notes:
            For normalized data (temperature in [-1,1], velocity scaled):
            - 'zeros' represents the mean/neutral state
            - After a few autoregressive steps, the model typically recovers reasonable predictions
            - This is a standard approach in autoregressive models when no prior is available
        """
        B, C_out, H, W = shape
        
        if mode == 'zeros':
            # Zeros represent the mean state for normalized data
            # Temperature: 0 = middle of [min, max] range
            # Velocity: 0 = no motion
            return torch.zeros(shape, device=device)
        
        elif mode == 'small_noise':
            # Small random noise to break symmetry
            # Useful if zeros cause numerical issues or mode collapse
            noise_scale = 0.01
            return torch.randn(shape, device=device) * noise_scale
        
        elif mode == 'from_conditioning':
            # Experimental: derive initial state from conditioning
            # Use velocity from interface as initial velocity estimate
            # Use a neutral temperature (zero = mean)
            if conditioning is None:
                print("⚠️ 'from_conditioning' mode requires conditioning tensor, falling back to zeros")
                return torch.zeros(shape, device=device)
            
            initial = torch.zeros(shape, device=device)
            
            # Assuming conditioning is [sdf, velx_interface, vely_interface]
            # and output is [velx, vely, temperature] or similar
            # Copy interface velocities to bulk velocity channels as initial guess
            if conditioning.shape[1] >= 3 and C_out >= 2:
                # Use interface velocity as initial bulk velocity estimate
                # This is physically motivated: bulk velocity near interface ≈ interface velocity
                initial[:, 0, :, :] = conditioning[:, 1, :, :]  # velx_interface -> velx
                initial[:, 1, :, :] = conditioning[:, 2, :, :]  # vely_interface -> vely
                # Temperature stays at zero (mean)
            
            return initial
        
        else:
            raise ValueError(f"Unknown initial state mode: {mode}. Use 'zeros', 'small_noise', or 'from_conditioning'")
    
    @torch.no_grad()
    def sample_trajectory(
        self,
        conditions: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        num_integration_steps: int = 50,
        solver: str = 'heun',
        guidance_scale: float = 1.0,
        stochastic_temperature: float = 0.0,
        initial_state_mode: str = 'zeros',
    ) -> torch.Tensor:
        """
        Generate a full trajectory autoregressively.
        
        Args:
            conditions: Conditioning for all timesteps [T, B, C_cond, H, W]
            initial_state: Initial state (output at t=0) [B, C_out, H, W].
                          If None, creates initial state based on initial_state_mode.
            num_integration_steps: Number of integration steps per frame
            solver: ODE solver - 'euler', 'heun', 'rk4', or 'midpoint'
            guidance_scale: Classifier-free guidance scale
            stochastic_temperature: Temperature for stochastic sampling (0.0 = deterministic)
            initial_state_mode: Mode for creating initial state when initial_state is None.
                               Options: 'zeros', 'small_noise', 'from_conditioning'
            
        Returns:
            Generated trajectory [T, B, C_out, H, W]
        """
        T = conditions.shape[0]
        device = conditions.device
        B = conditions.shape[1]
        C_cond = conditions.shape[2]
        H, W = conditions.shape[3], conditions.shape[4]
        
        # Determine output channels from first condition
        # For flow matching, this is typically same as num_target_channels
        C_out = C_cond  # Default assumption, may be overridden
        
        # Create initial state if not provided
        if initial_state is None:
            # Infer output shape from conditioning (same spatial dims, C_out channels)
            # Note: This assumes C_out = C_cond which may not always be true
            # Better to pass initial_state or use a model attribute
            shape = (B, C_out, H, W)
            prev_output = self.create_initial_state(
                shape=shape,
                device=device,
                mode=initial_state_mode,
                conditioning=conditions[0] if initial_state_mode == 'from_conditioning' else None
            )
        else:
            prev_output = initial_state
        
        trajectory = []
        
        for t in range(T):
            condition_t = conditions[t]  # [B, C_cond, H, W]
            
            # Generate output at timestep t
            output_t = self.sample(
                condition_t,
                prev_output,
                prev_output.shape,
                device,
                num_integration_steps,
                solver=solver,
                guidance_scale=guidance_scale,
                stochastic_temperature=stochastic_temperature,
            )
            
            trajectory.append(output_t)
            prev_output = output_t
        
        return torch.stack(trajectory, dim=0)


class ConditionalFlowMatchingARLightning(L.LightningModule):
    """
    PyTorch Lightning wrapper for Autoregressive Flow Matching model.
    
    Supports autoregressive training with teacher forcing and scheduled sampling:
    - Input: [conditioning_t, output_{t-1}] (ground truth or predicted previous state)
    - Target: output_t
    
    Training Modes:
    1. Teacher Forcing (scheduled_sampling.enabled=False):
       - Always uses ground truth previous state
       - Efficient parallel training
    
    2. Scheduled Sampling (scheduled_sampling.enabled=True):
       - Gradually transitions from teacher forcing to model predictions
       - Reduces exposure bias for better inference performance
    
    The model learns to predict the current state given the previous state,
    which naturally enforces temporal consistency.
    
    Args:
        model_cfg: Model configuration
        optim_cfg: Optimizer configuration
        scheduler_cfg: Learning rate scheduler configuration
        task_cfg: Task configuration defining channels
    """
    
    def __init__(
        self,
        model_cfg: DictConfig,
        optim_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        task_cfg: Optional[DictConfig] = None,
        normalization_stats: Optional[dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['normalization_stats'])
        
        # Store normalization statistics for accurate denormalization during logging
        self.normalization_stats = normalization_stats
        
        # Store task configuration
        self.task_cfg = task_cfg
        
        # Parse task config for channel indices
        if task_cfg is not None:
            self.conditioning_channels = list(task_cfg.get('conditioning_channels', [0]))
            self.target_channels = list(task_cfg.get('target_channels', [0]))
            self.task_name = task_cfg.get('name', 'unknown')
            
            # Check for noise configuration
            self.is_noisy_task = 'noisy' in self.task_name.lower()
            noise_cfg = task_cfg.get('noise_cfg', None)
            self.has_noise = noise_cfg is not None and noise_cfg.get('enabled', True)
            
            print(f"🎯 Task: {self.task_name}")
            print(f"   Conditioning channels: {self.conditioning_channels} ({task_cfg.get('conditioning_names', [])})")
            print(f"   Target channels: {self.target_channels} ({task_cfg.get('target_names', [])})")
            
            if self.has_noise:
                print(f"   🔊 Noise injection: ENABLED")
        else:
            self.conditioning_channels = [0]
            self.target_channels = [0]
            self.task_name = 'temperature_from_sdf'
            self.is_noisy_task = False
            self.has_noise = False
            print("⚠️  No task_cfg provided, defaulting to temperature_from_sdf task")
        
        # Compute channel counts
        self.num_conditioning_channels = len(self.conditioning_channels)
        self.num_target_channels = len(self.target_channels)
        
        # UNet configuration for autoregressive model
        # in_channels = num_target_channels (x_t) + num_conditioning_channels + num_target_channels (prev_output)
        # For Task 2: 3 + 3 + 3 = 9
        in_channels = self.num_target_channels + self.num_conditioning_channels + self.num_target_channels
        out_channels = self.num_target_channels
        
        print(f"\n🔄 Autoregressive Flow Matching Configuration:")
        print(f"   UNet in_channels: {in_channels} = {self.num_target_channels} (x_t) + {self.num_conditioning_channels} (cond) + {self.num_target_channels} (prev)")
        print(f"   UNet out_channels: {out_channels}")
        
        use_attention = model_cfg.get('use_attention', True)
        if isinstance(use_attention, bool):
            attention_type = 'bottleneck' if use_attention else 'none'
        else:
            attention_type = use_attention
        attention_type = model_cfg.get('attention_type', attention_type)

        # Initialize UNet
        unet = FlowMatchingUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=model_cfg.get('base_channels', 64),
            time_embed_dim=model_cfg.get('time_embed_dim', 320),
            num_res_blocks=model_cfg.get('num_res_blocks', 2),
            attention_type=attention_type,
            dropout=model_cfg.get('dropout', 0.0),
        )
        
        # Initialize autoregressive flow matching model
        self.flow_matching = ConditionalFlowMatchingAR(unet=unet)
        
        # =============================================================================
        # LOSS FUNCTION CONFIGURATION
        # =============================================================================
        # Primary loss
        self.loss_fn = nn.MSELoss()
        
        # Auxiliary losses to prevent blur
        loss_cfg = model_cfg.get('auxiliary_losses', {})
        
        # Spectral loss (preserves high-frequency details)
        self.use_spectral_loss = loss_cfg.get('spectral_enabled', False)
        if self.use_spectral_loss:
            self.spectral_loss = SpectralLoss(
                weight=loss_cfg.get('spectral_weight', 0.1),
                high_freq_weight=loss_cfg.get('spectral_high_freq_weight', 2.0),
                freq_threshold=loss_cfg.get('spectral_freq_threshold', 0.3),
            )
            print(f"📊 Spectral Loss: ENABLED (weight={loss_cfg.get('spectral_weight', 0.1)})")
        else:
            self.spectral_loss = None
            print(f"📊 Spectral Loss: DISABLED")
        
        # Gradient loss (preserves edges)
        self.use_gradient_loss = loss_cfg.get('gradient_enabled', False)
        if self.use_gradient_loss:
            self.gradient_loss = GradientLoss(
                weight=loss_cfg.get('gradient_weight', 0.1),
            )
            print(f"📐 Gradient Loss: ENABLED (weight={loss_cfg.get('gradient_weight', 0.1)})")
        else:
            self.gradient_loss = None
            print(f"📐 Gradient Loss: DISABLED")
        
        # -------------------------------------------------------------------------
        # PHYSICS-INFORMED LOSSES
        # -------------------------------------------------------------------------
        # Get downsample factor from data config (passed through normalization_stats)
        # Default to 1 (full resolution) if not available
        self.downsample_factor = 1
        if normalization_stats is not None:
            self.downsample_factor = normalization_stats.get('downsample_factor', 1)
        
        # 1. Divergence-Free Loss (Mass Conservation)
        # Uses MASSFLUX-based interface detection (matching physics_metrics_task2.py)
        self.use_divergence_loss = loss_cfg.get('divergence_enabled', False)
        if self.use_divergence_loss:
            self.divergence_loss = DivergenceLoss(
                weight=loss_cfg.get('divergence_weight', 0.1),
                downsample_factor=self.downsample_factor,
                massflux_threshold=loss_cfg.get('divergence_massflux_threshold', 1e-2),
                exclude_interface=loss_cfg.get('divergence_exclude_interface', True),
            )
            print(f"🌊 Divergence Loss: ENABLED (weight={loss_cfg.get('divergence_weight', 0.1)}, ds={self.downsample_factor}, massflux_threshold={loss_cfg.get('divergence_massflux_threshold', 1e-2)})")
        else:
            self.divergence_loss = None
            print(f"🌊 Divergence Loss: DISABLED")
        
        # 2. Vorticity Consistency Loss
        # Uses MASSFLUX-based interface detection (matching physics_metrics_task2.py)
        self.use_vorticity_loss = loss_cfg.get('vorticity_enabled', False)
        if self.use_vorticity_loss:
            self.vorticity_loss = VorticityLoss(
                weight=loss_cfg.get('vorticity_weight', 0.1),
                downsample_factor=self.downsample_factor,
                massflux_threshold=loss_cfg.get('vorticity_massflux_threshold', 1e-2),
                exclude_interface=loss_cfg.get('vorticity_exclude_interface', True),
            )
            print(f"🌀 Vorticity Loss: ENABLED (weight={loss_cfg.get('vorticity_weight', 0.1)}, massflux_threshold={loss_cfg.get('vorticity_massflux_threshold', 1e-2)})")
        else:
            self.vorticity_loss = None
            print(f"🌀 Vorticity Loss: DISABLED")
        
        # 3. Advection Consistency Loss (Velocity-Temperature Coupling)
        self.use_advection_loss = loss_cfg.get('advection_enabled', False)
        if self.use_advection_loss:
            self.advection_loss = AdvectionConsistencyLoss(
                weight=loss_cfg.get('advection_weight', 0.1),
                downsample_factor=self.downsample_factor,
                dt=loss_cfg.get('advection_dt', 0.001),
                interface_threshold=loss_cfg.get('advection_interface_threshold', 0.1),
                exclude_interface=loss_cfg.get('advection_exclude_interface', True),
                exclude_near_wall=loss_cfg.get('advection_exclude_near_wall', True),
                wall_rows=loss_cfg.get('advection_wall_rows', 4),
            )
            print(f"🔥 Advection Loss: ENABLED (weight={loss_cfg.get('advection_weight', 0.1)}, dt={loss_cfg.get('advection_dt', 0.001)})")
        else:
            self.advection_loss = None
            print(f"🔥 Advection Loss: DISABLED")
        
        # =============================================================================
        # INFERENCE CONFIGURATION
        # =============================================================================
        inference_cfg = model_cfg.get('inference', {})
        self.default_solver = inference_cfg.get('solver', 'heun')
        self.default_guidance_scale = inference_cfg.get('guidance_scale', 1.0)
        print(f"\n🔧 Default Inference Settings:")
        print(f"   Solver: {self.default_solver}")
        print(f"   Guidance scale: {self.default_guidance_scale}")
        
        # =============================================================================
        # RESIDUAL/DELTA PREDICTION MODE
        # =============================================================================
        # Instead of predicting absolute values, predict delta from previous state:
        #   delta_t = output_t - output_{t-1}
        #   output_t = output_{t-1} + delta_t
        #
        # Benefits:
        # - Errors are additive, not compounding multiplicatively
        # - Targets have smaller magnitude (easier to learn)
        # - Matches physics (slow thermal/velocity evolution)
        # - Reduces blur and energy diffusion in AR rollout
        self.residual_prediction = model_cfg.get('residual_prediction', False)
        if self.residual_prediction:
            print(f"\n🔄 Residual Prediction Mode: ENABLED")
            print(f"   Model predicts Δoutput = output_t - output_{{t-1}}")
            print(f"   Reconstruction: output_t = output_{{t-1}} + Δoutput")
        else:
            print(f"\n🔄 Residual Prediction Mode: DISABLED (absolute values)")
        
        # Temperature normalization parameters
        # Use computed stats if available, otherwise fall back to config values
        if normalization_stats is not None and 'temperature' in normalization_stats:
            temp_stats = normalization_stats['temperature']
            self.temp_min = temp_stats['min']
            self.temp_max = temp_stats['max']
            self.temp_range = self.temp_max - self.temp_min
            self.unified_velocity_scale = normalization_stats.get('unified_velocity_scale', 1.0)
            self.sdf_scale = normalization_stats.get('sdf', {}).get('scale', 1.0)
            print(f"📊 Using computed normalization stats for logging:")
            print(f"   Temperature: [{self.temp_min:.2f}, {self.temp_max:.2f}]°C")
            print(f"   Velocity scale: {self.unified_velocity_scale:.4f}")
        else:
            self.temp_min = model_cfg.get('temp_min', 55.0)
            self.temp_max = model_cfg.get('temp_max', 120.0)
            self.temp_range = self.temp_max - self.temp_min
            self.unified_velocity_scale = 1.0
            self.sdf_scale = 1.0
            print(f"⚠️  Using config normalization params (no stats provided)")
        
        # Flow matching parameters
        self.num_integration_steps = model_cfg.get('num_integration_steps', 50)
        
        # =============================================================================
        # SCHEDULED SAMPLING CONFIGURATION
        # =============================================================================
        ss_cfg = model_cfg.get('scheduled_sampling', {})
        self.scheduled_sampling_enabled = ss_cfg.get('enabled', False)
        
        if self.scheduled_sampling_enabled:
            self.ss_schedule_type = ss_cfg.get('schedule_type', 'linear')
            self.ss_warmup_epochs = ss_cfg.get('warmup_epochs', 5)
            self.ss_transition_epochs = ss_cfg.get('transition_epochs', 40)
            self.ss_min_teacher_ratio = ss_cfg.get('min_teacher_ratio', 0.0)
            self.ss_exponential_decay = ss_cfg.get('exponential_decay_rate', 0.95)
            self.ss_sigmoid_k = ss_cfg.get('sigmoid_k', 5.0)
            self.ss_sampling_steps = ss_cfg.get('sampling_steps', 20)
            
            print(f"\n📊 Scheduled Sampling: ENABLED")
            print(f"   Schedule type: {self.ss_schedule_type}")
            print(f"   Warmup epochs: {self.ss_warmup_epochs} (pure teacher forcing)")
            print(f"   Transition epochs: {self.ss_transition_epochs}")
            print(f"   Final teacher ratio: {self.ss_min_teacher_ratio:.1%}")
            print(f"   Sampling steps for predictions: {self.ss_sampling_steps}")
        else:
            print(f"\n📊 Scheduled Sampling: DISABLED (pure teacher forcing)")
        
        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.scheduler_cfg = scheduler_cfg
        
    def forward(self, x_t, condition, prev_output, t):
        return self.flow_matching(x_t, condition, prev_output, t)
    
    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        prev_output: torch.Tensor,
        shape: Tuple[int, ...],
        device: torch.device,
        num_integration_steps: Optional[int] = None,
        solver: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        stochastic_temperature: float = 0.0,
    ) -> torch.Tensor:
        """
        Generate samples with automatic residual reconstruction.
        
        This is the main inference method that handles:
        - ODE/SDE integration to sample from flow matching
        - Residual reconstruction if residual_prediction is enabled
        
        Args:
            condition: Current conditioning [B, C_cond, H, W]
            prev_output: Previous timestep output [B, C_out, H, W]
            shape: Output shape (B, C_out, H, W)
            device: Device for generation
            num_integration_steps: Steps (default from config)
            solver: ODE solver (default from config)
            guidance_scale: CFG scale (default from config)
            stochastic_temperature: Temperature for stochastic sampling (0.0 = deterministic ODE,
                                   >0 = SDE with noise for diverse trajectories)
            
        Returns:
            Generated samples [B, C_out, H, W] (absolute values)
        """
        # Use defaults from config if not specified
        if num_integration_steps is None:
            num_integration_steps = self.num_integration_steps
        if solver is None:
            solver = self.default_solver
        if guidance_scale is None:
            guidance_scale = self.default_guidance_scale
        
        # Sample from flow matching (outputs delta if residual mode, else absolute)
        raw_output = self.flow_matching.sample(
            condition=condition,
            prev_output=prev_output,
            shape=shape,
            device=device,
            num_integration_steps=num_integration_steps,
            solver=solver,
            guidance_scale=guidance_scale,
            stochastic_temperature=stochastic_temperature,
        )
        
        # Residual reconstruction: output = prev_output + delta
        if self.residual_prediction:
            output = prev_output + raw_output
        else:
            output = raw_output
        
        return output
    
    @torch.no_grad()
    def create_initial_state(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        mode: str = 'zeros',
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Create an initial state for autoregressive inference when ground truth is not available.
        
        This is useful for pure inference scenarios where we don't have access to the
        ground truth previous state (e.g., deploying on new data with only SDF and 
        interface velocity inputs).
        
        Args:
            shape: Output shape (B, C_out, H, W)
            device: Device for the tensor
            mode: Initialization mode:
                - 'zeros': All zeros (neutral state for normalized data)
                - 'small_noise': Small random noise around zero
                - 'from_conditioning': Derive from conditioning (use interface vel as bulk vel)
            conditioning: Optional conditioning tensor [B, C_cond, H, W] for 'from_conditioning' mode
            
        Returns:
            Initial state tensor [B, C_out, H, W]
            
        Notes:
            For normalized data (temperature in [-1,1], velocity scaled):
            - 'zeros' represents the mean/neutral state
            - After a few autoregressive steps, the model typically recovers reasonable predictions
            - 'from_conditioning' uses interface velocity as initial bulk velocity estimate
        """
        return self.flow_matching.create_initial_state(
            shape=shape,
            device=device,
            mode=mode,
            conditioning=conditioning
        )
    
    def denormalize_temperature(self, temperature_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize temperature from [-1, 1] to Celsius."""
        return (temperature_norm + 1.0) / 2.0 * self.temp_range + self.temp_min
    
    def _extract_channels(self, tensor: torch.Tensor, channel_indices: list) -> torch.Tensor:
        """Extract specific channels from a tensor [B, C, H, W]."""
        return tensor[:, channel_indices, :, :]
    
    def get_teacher_forcing_ratio(self) -> float:
        """
        Compute the teacher forcing ratio based on current epoch and schedule.
        
        Returns:
            float: Probability of using ground truth (1.0 = all teacher forcing,
                   0.0 = all model predictions)
        """
        if not self.scheduled_sampling_enabled:
            return 1.0  # Pure teacher forcing
        
        current_epoch = self.current_epoch
        
        # During warmup: pure teacher forcing
        if current_epoch < self.ss_warmup_epochs:
            return 1.0
        
        # Epochs into the transition phase
        transition_epoch = current_epoch - self.ss_warmup_epochs
        
        # After transition: minimum teacher ratio
        if transition_epoch >= self.ss_transition_epochs:
            return self.ss_min_teacher_ratio
        
        # Progress through transition (0 to 1)
        progress = transition_epoch / self.ss_transition_epochs
        
        if self.ss_schedule_type == 'linear':
            # Linear decay: ratio = 1 - progress * (1 - min_ratio)
            teacher_ratio = 1.0 - progress * (1.0 - self.ss_min_teacher_ratio)
        
        elif self.ss_schedule_type == 'exponential':
            # Exponential decay: ratio = max(min_ratio, decay^epoch)
            teacher_ratio = max(
                self.ss_min_teacher_ratio,
                self.ss_exponential_decay ** transition_epoch
            )
        
        elif self.ss_schedule_type == 'inverse_sigmoid':
            # Inverse sigmoid: smooth S-curve transition
            # Starts slow, accelerates in middle, slows at end
            k = self.ss_sigmoid_k
            # Map progress to sigmoid input centered at 0.5
            x = k * (progress - 0.5)
            sigmoid_val = 1.0 / (1.0 + math.exp(-x))
            # Map from [sigmoid(-k/2), sigmoid(k/2)] to [1, min_ratio]
            teacher_ratio = 1.0 - sigmoid_val * (1.0 - self.ss_min_teacher_ratio)
        
        else:
            # Default to linear
            teacher_ratio = 1.0 - progress * (1.0 - self.ss_min_teacher_ratio)
        
        return max(self.ss_min_teacher_ratio, min(1.0, teacher_ratio))
    
    @torch.no_grad()
    def _generate_predicted_prev_output(
        self,
        conditioning_t_minus_1: torch.Tensor,
        output_t_minus_2: torch.Tensor,
        target_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Generate model's prediction for output at t-1 (for scheduled sampling).
        
        Uses self.sample() which handles residual reconstruction automatically.
        
        Args:
            conditioning_t_minus_1: Conditioning at timestep t-1 [B, C_cond, H, W]
            output_t_minus_2: Ground truth output at t-2 [B, C_out, H, W]
            target_shape: Shape of the output (B, C_out, H, W)
            
        Returns:
            Predicted output at t-1 [B, C_out, H, W] (absolute values)
        """
        # Use self.sample() which handles residual reconstruction
        return self.sample(
            condition=conditioning_t_minus_1,
            prev_output=output_t_minus_2,
            shape=target_shape,
            device=conditioning_t_minus_1.device,
            num_integration_steps=self.ss_sampling_steps,
            solver='euler',  # Use fast euler for scheduled sampling
            guidance_scale=1.0,  # No guidance during training
        )
    
    def training_step(self, batch, batch_idx):
        """
        Training step with teacher forcing or scheduled sampling.
        
        Teacher Forcing Mode:
            Batch: (inp_data_t, prev_output_gt, out_data_t)
            Uses ground truth prev_output for all samples.
        
        Scheduled Sampling Mode:
            Batch: (inp_data_t, prev_output_gt, out_data_t, 
                    conditioning_t_minus_1, output_t_minus_2)
            Randomly decides per-sample whether to use ground truth or 
            model's prediction for prev_output based on teacher_ratio.
        """
        # Determine if we're using scheduled sampling
        use_scheduled_sampling = (
            self.scheduled_sampling_enabled and 
            len(batch) >= 5  # Extra context provided
        )
        
        # Extract data from batch based on mode
        if use_scheduled_sampling:
            (inp_data_t, prev_output_raw, out_data_t,
             conditioning_t_minus_1_raw, output_t_minus_2_raw) = batch
        else:
            # Standard teacher forcing
            inp_data_t, prev_output_raw, out_data_t = batch
        
        # Extract conditioning and target channels
        conditioning = self._extract_channels(inp_data_t, self.conditioning_channels)  # [B, C_cond, H, W]
        target = self._extract_channels(out_data_t, self.target_channels)  # [B, C_out, H, W]
        prev_output_gt = self._extract_channels(prev_output_raw, self.target_channels)  # [B, C_out, H, W]
        
        batch_size = conditioning.shape[0]
        device = conditioning.device
        
        # Determine prev_output to use (teacher forcing vs model prediction)
        if use_scheduled_sampling:
            teacher_ratio = self.get_teacher_forcing_ratio()
            
            # Log the current teacher ratio
            self.log('teacher_ratio', teacher_ratio, on_step=False, on_epoch=True, prog_bar=True)
            
            if teacher_ratio < 1.0:
                # Extract extra context for scheduled sampling
                conditioning_t_minus_1 = self._extract_channels(
                    conditioning_t_minus_1_raw, self.conditioning_channels
                )
                output_t_minus_2 = self._extract_channels(
                    output_t_minus_2_raw, self.target_channels
                )
                
                # Decide which samples use predictions vs ground truth
                use_teacher = torch.rand(batch_size, device=device) < teacher_ratio
                
                # Generate predictions for samples that need them
                num_predictions = (~use_teacher).sum().item()
                
                if num_predictions > 0:
                    # Get indices of samples needing predictions
                    pred_indices = torch.where(~use_teacher)[0]
                    
                    # Generate predictions for those samples
                    predicted_prev_output = self._generate_predicted_prev_output(
                        conditioning_t_minus_1[pred_indices],
                        output_t_minus_2[pred_indices],
                        (num_predictions, self.num_target_channels, target.shape[2], target.shape[3])
                    )
                    
                    # Mix ground truth and predictions
                    prev_output = prev_output_gt.clone()
                    prev_output[pred_indices] = predicted_prev_output
                    
                    # Log prediction usage stats
                    self.log('pct_predictions', 100.0 * num_predictions / batch_size, 
                            on_step=False, on_epoch=True, prog_bar=False)
                else:
                    prev_output = prev_output_gt
            else:
                # Pure teacher forcing this epoch
                prev_output = prev_output_gt
        else:
            # Pure teacher forcing (scheduled sampling disabled)
            prev_output = prev_output_gt
        
        # Sample random time values
        t = torch.rand(batch_size, device=device)
        
        # =======================================================================
        # RESIDUAL PREDICTION MODE
        # =======================================================================
        # If enabled, train to predict delta instead of absolute values
        # delta_target = target - prev_output
        if self.residual_prediction:
            # Compute delta (temporal change)
            delta_target = target - prev_output
            flow_target = delta_target
            
            # Log delta statistics for monitoring
            if batch_idx % 100 == 0:
                delta_mean = delta_target.abs().mean()
                delta_max = delta_target.abs().max()
                self.log('delta_mean', delta_mean, on_step=False, on_epoch=True)
                self.log('delta_max', delta_max, on_step=False, on_epoch=True)
        else:
            # Standard absolute prediction
            flow_target = target
        
        # Sample noise (source distribution)
        x_0 = torch.randn_like(flow_target)
        
        # Compute conditional flow path (now towards delta or absolute)
        x_t, velocity_target = self.flow_matching.compute_conditional_flow(x_0, flow_target, t)
        
        # Predict velocity field with autoregressive conditioning
        velocity_pred = self.flow_matching(x_t, conditioning, prev_output, t)
        
        # Compute primary MSE loss on velocity
        mse_loss = self.loss_fn(velocity_pred, velocity_target)
        
        # Compute auxiliary losses
        total_loss = mse_loss
        
        if self.use_spectral_loss and self.spectral_loss is not None:
            spec_loss = self.spectral_loss(velocity_pred, velocity_target)
            total_loss = total_loss + spec_loss
            self.log('train_spectral_loss', spec_loss, on_step=False, on_epoch=True)
        
        if self.use_gradient_loss and self.gradient_loss is not None:
            grad_loss = self.gradient_loss(velocity_pred, velocity_target)
            total_loss = total_loss + grad_loss
            self.log('train_gradient_loss', grad_loss, on_step=False, on_epoch=True)
        
        # =====================================================================
        # PHYSICS-INFORMED LOSSES
        # =====================================================================
        # Physics losses are applied on the IMPLICIT predicted output:
        #   predicted_output = x_0 + velocity_pred
        # This represents what the model would generate at t=1 (end of flow)
        #
        # For Task 2 (velocity_from_interface):
        #   target_channels = [1, 2, 0] → [velx, vely, temperature] in output
        #   Channel 0 = velx, Channel 1 = vely, Channel 2 = temperature
        #   SDF is in conditioning channel 0
        # =====================================================================
        
        has_physics_loss = (self.use_divergence_loss or 
                           self.use_vorticity_loss or 
                           self.use_advection_loss)
        
        if has_physics_loss:
            # Compute implicit predicted output at t=1 (what model generates)
            # output_pred = x_0 + velocity_pred (the target of flow matching)
            # NOTE: This is the PREDICTED PHYSICAL velocity/temperature, not the flow matching velocity
            output_pred = x_0 + velocity_pred  # [B, C_out, H, W]
            
            # Extract from conditioning for interface masking
            # Conditioning channels: [sdf, velx_interface, vely_interface]
            sdf = conditioning[:, 0, :, :]  # [B, H, W] - for temperature (advection loss)
            velx_interface = conditioning[:, 1, :, :]  # [B, H, W] - for velocity losses
            vely_interface = conditioning[:, 2, :, :]  # [B, H, W] - for velocity losses
            
            # Extract velocity channels from predictions and targets
            # Target channels order: [velx, vely, temperature] (indices 0, 1, 2)
            pred_velx = output_pred[:, 0, :, :]  # [B, H, W]
            pred_vely = output_pred[:, 1, :, :]  # [B, H, W]
            target_velx = flow_target[:, 0, :, :]  # [B, H, W]
            target_vely = flow_target[:, 1, :, :]  # [B, H, W]
            
            # 1. Divergence-Free Loss (Mass Conservation)
            # Uses MASSFLUX-based interface detection (matching physics_metrics_task2.py)
            if self.use_divergence_loss and self.divergence_loss is not None:
                div_loss = self.divergence_loss(
                    pred_velx=pred_velx,
                    pred_vely=pred_vely,
                    velx_interface=velx_interface,
                    vely_interface=vely_interface,
                )
                total_loss = total_loss + div_loss
                self.log('train_divergence_loss', div_loss, on_step=False, on_epoch=True)
            
            # 2. Vorticity Consistency Loss
            # Uses MASSFLUX-based interface detection (matching physics_metrics_task2.py)
            if self.use_vorticity_loss and self.vorticity_loss is not None:
                vort_loss = self.vorticity_loss(
                    pred_velx=pred_velx,
                    pred_vely=pred_vely,
                    target_velx=target_velx,
                    target_vely=target_vely,
                    velx_interface=velx_interface,
                    vely_interface=vely_interface,
                )
                total_loss = total_loss + vort_loss
                self.log('train_vorticity_loss', vort_loss, on_step=False, on_epoch=True)
            
            # 3. Advection Consistency Loss
            # Uses SDF-based interface detection (appropriate for temperature metrics)
            if self.use_advection_loss and self.advection_loss is not None:
                # Extract temperature from predictions and previous output
                pred_temp = output_pred[:, 2, :, :]  # [B, H, W]
                
                # Previous output channels: [velx_{t-1}, vely_{t-1}, temp_{t-1}]
                prev_velx = prev_output[:, 0, :, :]  # [B, H, W]
                prev_vely = prev_output[:, 1, :, :]  # [B, H, W]
                prev_temp = prev_output[:, 2, :, :]  # [B, H, W]
                
                adv_loss = self.advection_loss(
                    pred_temp=pred_temp,
                    prev_temp=prev_temp,
                    prev_velx=prev_velx,
                    prev_vely=prev_vely,
                    sdf=sdf,
                )
                total_loss = total_loss + adv_loss
                self.log('train_advection_loss', adv_loss, on_step=False, on_epoch=True)
        
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mse_loss', mse_loss, on_step=False, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step."""
        # Determine validation type
        is_clean_val = (dataloader_idx == 0) if self.has_noise else True
        val_prefix = "val_clean" if is_clean_val else "val_noisy"
        
        # Extract data
        inp_data_t, prev_output_raw, out_data_t = batch
        
        # Extract channels
        conditioning = self._extract_channels(inp_data_t, self.conditioning_channels)
        target = self._extract_channels(out_data_t, self.target_channels)
        prev_output = self._extract_channels(prev_output_raw, self.target_channels)
        
        batch_size = conditioning.shape[0]
        device = conditioning.device
        
        # Sample random time values
        t = torch.rand(batch_size, device=device)
        
        # Residual prediction mode
        if self.residual_prediction:
            delta_target = target - prev_output
            flow_target = delta_target
        else:
            flow_target = target
        
        # Sample noise
        x_0 = torch.randn_like(flow_target)
        
        # Compute conditional flow path
        x_t, velocity_target = self.flow_matching.compute_conditional_flow(x_0, flow_target, t)
        
        # Predict velocity
        velocity_pred = self.flow_matching(x_t, conditioning, prev_output, t)
        
        # Compute loss
        loss = self.loss_fn(velocity_pred, velocity_target)
        
        # Log losses
        self.log(f'{val_prefix}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        
        if is_clean_val:
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        
        # Generate samples and log statistics
        if batch_idx == 0 and is_clean_val:
            with torch.no_grad():
                num_samples = min(4, batch_size)
                
                # Generate samples with teacher forcing (ground truth prev_output)
                # Uses self.sample() which handles residual reconstruction automatically
                samples = self.sample(
                    condition=conditioning[:num_samples],
                    prev_output=prev_output[:num_samples],
                    shape=(num_samples, self.num_target_channels, target.shape[2], target.shape[3]),
                    device=device,
                    num_integration_steps=self.num_integration_steps,
                )
                
                # Log statistics
                sample_mean = samples.mean()
                sample_std = samples.std()
                target_mean = target[:num_samples].mean()
                target_std = target[:num_samples].std()
                
                self.log(f'{val_prefix}_sample_mean_norm', sample_mean, on_step=False, on_epoch=True, add_dataloader_idx=False)
                self.log(f'{val_prefix}_sample_std_norm', sample_std, on_step=False, on_epoch=True, add_dataloader_idx=False)
                self.log(f'{val_prefix}_target_mean_norm', target_mean, on_step=False, on_epoch=True, add_dataloader_idx=False)
                self.log(f'{val_prefix}_target_std_norm', target_std, on_step=False, on_epoch=True, add_dataloader_idx=False)
                
                # Log temperature statistics if applicable
                if self.task_cfg is not None and 'temperature' in self.task_cfg.get('target_names', []):
                    temp_idx = list(self.task_cfg.get('target_names', [])).index('temperature')
                    samples_temp = samples[:, temp_idx:temp_idx+1, :, :]
                    target_temp = target[:num_samples, temp_idx:temp_idx+1, :, :]
                    
                    samples_celsius = self.denormalize_temperature(samples_temp)
                    target_celsius = self.denormalize_temperature(target_temp)
                    
                    self.log(f'{val_prefix}_pred_temp_min_C', samples_celsius.min(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                    self.log(f'{val_prefix}_pred_temp_max_C', samples_celsius.max(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                    self.log(f'{val_prefix}_target_temp_min_C', target_celsius.min(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                    self.log(f'{val_prefix}_target_temp_max_C', target_celsius.max(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        
        return loss
    
    def configure_optimizers(self):
        # Configure optimizer
        if self.optim_cfg.name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.optim_cfg.get('lr', 1e-3),
                weight_decay=self.optim_cfg.get('weight_decay', 0.0)
            )
        elif self.optim_cfg.name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.optim_cfg.get('lr', 1e-3),
                weight_decay=self.optim_cfg.get('weight_decay', 1e-2)
            )
        elif self.optim_cfg.name.lower() == 'lion':
            try:
                from lion_pytorch import Lion
                optimizer = Lion(
                    self.parameters(),
                    lr=self.optim_cfg.get('lr', 1e-4),
                    weight_decay=self.optim_cfg.get('weight_decay', 1e-2)
                )
            except ImportError:
                print("Lion optimizer not available, falling back to AdamW")
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=self.optim_cfg.get('lr', 1e-3),
                    weight_decay=self.optim_cfg.get('weight_decay', 1e-2)
                )
        else:
            raise ValueError(f"Unknown optimizer: {self.optim_cfg.name}")
        
        # Configure scheduler
        if self.scheduler_cfg.name.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.optim_cfg.get('lr', 1e-3) * 0.01
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        elif self.scheduler_cfg.name.lower() == 'cosine_warmup':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.scheduler_cfg.get('T_0', 10),
                T_mult=self.scheduler_cfg.get('T_mult', 2),
                eta_min=self.optim_cfg.get('lr', 1e-3) * 0.01
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        else:
            return optimizer
