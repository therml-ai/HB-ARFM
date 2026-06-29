"""
This module contains dataset classes for BubbleML dataset.
Classes:
    BulkFlow: Dataset class for predicting bulk liquid velocity and temperature
              from bubble position and interface velocity.

Normalization Scheme:
- Temperature: Tanh normalization → [-1, 1] range
- Velocity (velx, vely, velx_interface, vely_interface): Unified shared scale normalization
- SDF: Zero-preserving normalization (preserves interface at zero)
"""
from typing import List, Optional, Tuple, Dict, Any
import json
import re
import os

import numpy as np
import h5py as h5
import torch
from torch.utils.data import Dataset

from bubblefusion.utils.noise import create_noise_model


def compute_normalization_stats(
    filenames: List[str],
    start_time: int = 0,
    verbose: bool = True,
    velocity_percentile: float = 99.0
) -> Dict[str, Dict[str, float]]:
    """
    Compute normalization statistics from all training files.
    
    This function scans all training files to compute:
    - Temperature: min/max for tanh normalization
    - Velocity fields: percentile-based unified scale for all velocity components
    - SDF: max absolute value for zero-preserving normalization
    
    Args:
        filenames: List of HDF5 file paths
        start_time: Starting timestep (to skip initial transients)
        velocity_percentile: Percentile to use for velocity scale (default 99.0).
                            Using percentile instead of max avoids extreme outliers
                            dominating the normalization, which would compress most
                            velocity values to near-zero.
        
    Returns:
        Dictionary with normalization parameters for each field
    """
    if verbose:
        print("\n📊 Computing normalization statistics from training data...")
        print(f"   Files: {len(filenames)}")
        print(f"   Start time: {start_time}")
    
    # Initialize tracking variables
    temp_min, temp_max = float('inf'), float('-inf')
    sdf_min, sdf_max = float('inf'), float('-inf')
    
    # Track velocity statistics for percentile-based normalization
    # We collect sampled absolute velocity values to compute percentiles
    vel_abs_samples = []  # Will store sampled |velocity| values
    vel_max = 0.0  # Also track max for reference
    
    # Also track interface velocity if available
    has_interface_velocity = False
    
    for i, filename in enumerate(filenames):
        if verbose:
            print(f"   Scanning [{i+1}/{len(filenames)}]: {os.path.basename(filename)}")
        
        with h5.File(filename, 'r') as f:
            # Get data from start_time onwards
            temp = f['temperature'][start_time:]
            velx = f['velx'][start_time:]
            vely = f['vely'][start_time:]
            sdf = f['dfun'][start_time:]
            
            # Update temperature bounds
            temp_min = min(temp_min, float(np.min(temp)))
            temp_max = max(temp_max, float(np.max(temp)))
            
            # Update SDF bounds
            file_sdf_min = float(np.min(sdf))
            file_sdf_max = float(np.max(sdf))
            
            # Flag files with unusual SDF values
            if verbose and (file_sdf_max > 100 or file_sdf_min < -100):
                print(f"      ⚠️  Unusual SDF range: [{file_sdf_min:.2f}, {file_sdf_max:.2f}]")
            
            sdf_min = min(sdf_min, file_sdf_min)
            sdf_max = max(sdf_max, file_sdf_max)
            
            # Track velocity statistics (bulk velocity)
            # Update max for reference
            vel_max = max(vel_max, abs(float(np.min(velx))), abs(float(np.max(velx))),
                         abs(float(np.min(vely))), abs(float(np.max(vely))))
            
            # Sample velocity magnitudes for percentile computation
            # Sample every 10th timestep and subsample spatially to keep memory manageable
            num_timesteps = velx.shape[0]
            sample_step = max(1, num_timesteps // 20)  # ~20 timesteps per file
            for t in range(0, num_timesteps, sample_step):
                # Compute velocity magnitude
                vel_mag = np.sqrt(velx[t]**2 + vely[t]**2)
                # Subsample spatially (every 4th pixel in each dimension)
                vel_mag_subsampled = vel_mag[::4, ::4].flatten()
                vel_abs_samples.extend(vel_mag_subsampled.tolist())
            
            # Check for interface velocity computation capability
            has_massflux = "massflux" in f.keys()
            fluid_params_file = filename.replace(".hdf5", ".json")
            has_params = os.path.exists(fluid_params_file)
            
            if has_massflux and has_params:
                has_interface_velocity = True
                # Compute actual interface velocities to get accurate statistics
                # Interface velocity = v_bulk * interface_region + (massflux/rho_gas) * normal
                # We need to sample timesteps to compute the actual interface velocity values
                try:
                    with open(fluid_params_file, 'r') as pf:
                        params = json.load(pf)
                    rho_gas = params.get('rhogas', 1.0)
                    # Get grid spacing for normal computation
                    sdf_shape = f['dfun'].shape
                    dy = (params["y_max"] - params["y_min"]) / sdf_shape[1]
                    dx = (params["x_max"] - params["x_min"]) / sdf_shape[2]
                    
                    # Sample timesteps to compute interface velocity statistics
                    # (computing for all timesteps would be too slow)
                    num_timesteps = sdf_shape[0] - start_time
                    num_samples = min(50, num_timesteps)  # Sample up to 50 timesteps
                    if num_samples > 0:
                        sample_indices = np.linspace(start_time, sdf_shape[0]-1, num_samples, dtype=int)
                        
                        for t_idx in sample_indices:
                            sdf_t = f['dfun'][t_idx]
                            massflux_t = f['massflux'][t_idx]
                            velx_t = f['velx'][t_idx]
                            vely_t = f['vely'][t_idx]
                            
                            # Compute interface region and normals
                            interface_mask = massflux_t != 0
                            if not np.any(interface_mask):
                                continue
                            
                            # Compute gradient for normals
                            norm_y, norm_x = np.gradient(sdf_t, dy, dx)
                            norm_mag = np.sqrt(norm_x**2 + norm_y**2) + 1e-8
                            norm_x = norm_x / norm_mag
                            norm_y = norm_y / norm_mag
                            
                            # Compute interface velocity
                            mflux_contrib = massflux_t / rho_gas
                            velx_int = velx_t * interface_mask + mflux_contrib * norm_x
                            vely_int = vely_t * interface_mask + mflux_contrib * norm_y
                            
                            # Track interface velocity statistics
                            velx_int_masked = velx_int[interface_mask]
                            vely_int_masked = vely_int[interface_mask]
                            
                            if len(velx_int_masked) > 0:
                                # Update max
                                vel_max = max(vel_max,
                                             abs(float(np.min(velx_int_masked))),
                                             abs(float(np.max(velx_int_masked))),
                                             abs(float(np.min(vely_int_masked))),
                                             abs(float(np.max(vely_int_masked))))
                                # Add interface velocity magnitudes to samples
                                vel_int_mag = np.sqrt(velx_int_masked**2 + vely_int_masked**2)
                                vel_abs_samples.extend(vel_int_mag.tolist())
                except (FileNotFoundError, json.JSONDecodeError, KeyError):
                    pass
    
    # Compute unified velocity scale from all velocity extremes
    # Compute unified velocity scale using percentile (avoids outlier domination)
    if len(vel_abs_samples) > 0:
        vel_abs_array = np.array(vel_abs_samples)
        unified_velocity_scale = float(np.percentile(vel_abs_array, velocity_percentile))
        # Ensure scale is at least some minimum to avoid division issues
        unified_velocity_scale = max(unified_velocity_scale, 0.1)
        
        if verbose:
            print(f"\n   📊 Velocity scale computation:")
            print(f"      Samples collected: {len(vel_abs_samples):,}")
            print(f"      Max velocity:      {vel_max:.4f}")
            print(f"      {velocity_percentile}th percentile: {unified_velocity_scale:.4f}")
            print(f"      Using {velocity_percentile}th percentile as scale (avoids outlier domination)")
    else:
        unified_velocity_scale = 1.0
    
    # Compute SDF scale (zero-preserving)
    sdf_scale = max(abs(sdf_min), abs(sdf_max))
    
    # Build normalization stats dictionary
    stats = {
        'temperature': {
            'min': temp_min,
            'max': temp_max,
            'center': (temp_min + temp_max) / 2,
            'half_range': (temp_max - temp_min) / 2,
            'method': 'tanh'  # Maps to [-1, 1]
        },
        'velx': {
            'scale': unified_velocity_scale,
            'method': 'scale'  # x_norm = x / scale
        },
        'vely': {
            'scale': unified_velocity_scale,
            'method': 'scale'
        },
        'velx_interface': {
            'scale': unified_velocity_scale,
            'method': 'scale'
        },
        'vely_interface': {
            'scale': unified_velocity_scale,
            'method': 'scale'
        },
        'sdf': {
            'scale': sdf_scale,
            'method': 'scale'  # Zero-preserving: x_norm = x / scale
        },
        # Store unified scale for reference
        'unified_velocity_scale': unified_velocity_scale,
        'velocity_max': vel_max,  # Store max for reference
        'velocity_percentile_used': velocity_percentile,
        'has_interface_velocity': has_interface_velocity
    }
    
    if verbose:
        print(f"\n   📈 Normalization Statistics:")
        print(f"      Temperature: [{temp_min:.2f}, {temp_max:.2f}]°C → tanh to [-1, 1]")
        print(f"      Unified velocity scale: {unified_velocity_scale:.4f} ({velocity_percentile}th percentile)")
        print(f"      Velocity max (for reference): {vel_max:.4f}")
        print(f"      SDF: [{sdf_min:.4f}, {sdf_max:.4f}] → scale={sdf_scale:.4f}")
        print(f"      Interface velocity available: {has_interface_velocity}")
    
    return stats


class NormalizationHelper:
    """Helper class for normalization and denormalization operations.
    
    Args:
        stats: Dictionary with normalization parameters from compute_normalization_stats()
        norm_mode: 'all' (default, normalize everything), 'none' (skip all normalization),
                   or 'temperature_only' (normalize temperature, leave velocity/sdf raw)
    """
    
    VALID_MODES = ('none', 'all', 'temperature_only')
    
    def __init__(self, stats: Dict[str, Dict[str, float]], norm_mode: str = 'all'):
        if norm_mode not in self.VALID_MODES:
            raise ValueError(f"norm_mode must be one of {self.VALID_MODES}, got '{norm_mode}'")
        self.stats = stats
        self.norm_mode = norm_mode
        
    def normalize_temperature(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize temperature using tanh normalization to [-1, 1]."""
        if self.norm_mode == 'none':
            return x
        t_min = self.stats['temperature']['min']
        t_max = self.stats['temperature']['max']
        t_range = t_max - t_min
        if t_range == 0:
            return torch.zeros_like(x)
        return 2.0 * (x - t_min) / t_range - 1.0
    
    def denormalize_temperature(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize temperature from [-1, 1] back to original range."""
        if self.norm_mode == 'none':
            return x_norm
        t_min = self.stats['temperature']['min']
        t_max = self.stats['temperature']['max']
        t_range = t_max - t_min
        return (x_norm + 1.0) / 2.0 * t_range + t_min
    
    def normalize_velocity(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize velocity using unified scale (zero-centered)."""
        if self.norm_mode in ('none', 'temperature_only'):
            return x
        scale = self.stats['velx']['scale']
        if scale == 0:
            return torch.zeros_like(x)
        return x / scale
    
    def denormalize_velocity(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize velocity from scaled value back to original."""
        if self.norm_mode in ('none', 'temperature_only'):
            return x_norm
        scale = self.stats['velx']['scale']
        return x_norm * scale
    
    def normalize_sdf(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize SDF using zero-preserving scale."""
        if self.norm_mode in ('none', 'temperature_only'):
            return x
        scale = self.stats['sdf']['scale']
        if scale == 0:
            return torch.zeros_like(x)
        return x / scale
    
    def denormalize_sdf(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize SDF from scaled value back to original."""
        if self.norm_mode in ('none', 'temperature_only'):
            return x_norm
        scale = self.stats['sdf']['scale']
        return x_norm * scale
    
    def normalize_field(self, x: torch.Tensor, field_name: str) -> torch.Tensor:
        """Normalize a field by name."""
        if field_name == 'temperature':
            return self.normalize_temperature(x)
        elif field_name in ['velx', 'vely', 'velx_interface', 'vely_interface']:
            return self.normalize_velocity(x)
        elif field_name in ['sdf', 'dfun']:
            return self.normalize_sdf(x)
        else:
            return x
    
    def denormalize_field(self, x_norm: torch.Tensor, field_name: str) -> torch.Tensor:
        """Denormalize a field by name."""
        if field_name == 'temperature':
            return self.denormalize_temperature(x_norm)
        elif field_name in ['velx', 'vely', 'velx_interface', 'vely_interface']:
            return self.denormalize_velocity(x_norm)
        elif field_name in ['sdf', 'dfun']:
            return self.denormalize_sdf(x_norm)
        else:
            return x_norm
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get the normalization statistics dictionary."""
        return self.stats

class BulkFlow(Dataset):
    """
    Dataset class for predicting bulk liquid velocity and temperature
    from bubble position and interface velocity.
    
    Normalization (based on NORMALIZATION_REQUIREMENTS.md):
    - Temperature: Tanh normalization to [-1, 1]
    - Velocity (velx, vely, velx_interface, vely_interface): Unified shared scale
    - SDF: Zero-preserving normalization (preserves interface at zero)
    """

    def __init__(
        self,
        filenames: List[str],
        output_fields: Optional[List[str]] = None,
        start_time: int = 0,
        normalization_stats: Optional[Dict[str, Dict[str, float]]] = None,
        return_wall_temp: bool = False,
        noise_cfg: Optional[Dict[str, Any]] = None,
        downsample_factor: int = 1,
        norm_mode: str = 'all',
        # Legacy parameter - kept for backward compatibility but ignored
        normalize_temperature: bool = True,
    ):
        super().__init__()
        self.filenames = filenames
        self.return_wall_temp = return_wall_temp
        self.downsample_factor = downsample_factor
        self.norm_mode = norm_mode
        
        # Initialize noise model for Task 3 (noisy optical flow simulation)
        # Supports both simple Gaussian noise and complex optical flow noise
        self.noise_model = create_noise_model(noise_cfg)
        if noise_cfg is not None and noise_cfg.get('enabled', True):
            noise_type = noise_cfg.get('noise_type', 'optical_flow')
            print(f"\n🔊 Noise Model Enabled (type: {noise_type}):")
            print(f"   {self.noise_model}")

        if output_fields is not None:
            self.output_fields = output_fields
        else:
            self.output_fields = ["temperature", "velx", "vely"]
        
        # Log downsampling configuration
        if self.downsample_factor > 1:
            print(f"\n📐 Downsampling Enabled:")
            print(f"   Factor: {self.downsample_factor}x")
            print(f"   Example: 512x512 → {512 // self.downsample_factor}x{512 // self.downsample_factor}")

        self.start_time = start_time
        self.data = [h5.File(filename, "r") for filename in filenames]
        self.num_trajs = []
        self.traj_lens = []  # Full trajectory lengths (before applying start_time)

        for h5_file in self.data:
            self.num_trajs.append(1)
            total_timesteps = h5_file["dfun"].shape[0]
            self.traj_lens.append(total_timesteps)
        
        # Report start_time usage
        if self.start_time > 0:
            print(f"\n⏭️  start_time = {self.start_time}")
            print(f"   Only using timesteps from {self.start_time} onwards")
            total_available = sum(max(0, traj_len - self.start_time) for traj_len in self.traj_lens)
            total_possible = sum(self.traj_lens)
            print(f"   Available samples: {total_available} / {total_possible} timesteps")
        
        # Extract wall temperatures from filenames
        self.wall_temps = []
        for filename in filenames:
            wall_temp = self._extract_wall_temp_from_filename(filename)
            self.wall_temps.append(wall_temp)
            if self.return_wall_temp:
                print(f"  Extracted Twall={wall_temp}°C from {os.path.basename(filename)}")

        # Try to load fluid params (JSON files) and check for massflux field
        self.fluid_params_files = [fname.replace(".hdf5", ".json") for fname in filenames]
        self.fluid_params = []
        self.has_interface_velocity = []  # Track which files can calculate interface velocity
        
        for i, (h5_file, fluid_params_file) in enumerate(zip(self.data, self.fluid_params_files)):
            # Check if massflux field exists in HDF5 file
            has_massflux = "massflux" in h5_file.keys()
            
            # Try to load fluid params JSON
            fluid_params = None
            if has_massflux:
                try:
                    with open(fluid_params_file, "r", encoding="utf-8") as f:
                        fluid_params = json.load(f)
                    print(f"✓ Loaded fluid params for file {i}: {filenames[i]}")
                except FileNotFoundError:
                    print(f"⚠️  Warning: massflux exists but JSON file not found for {filenames[i]}")
            
            self.fluid_params.append(fluid_params)
            
            # Can only calculate interface velocity if both massflux and params exist
            can_calc_interface = has_massflux and (fluid_params is not None)
            self.has_interface_velocity.append(can_calc_interface)
            
            if not has_massflux:
                print(f"ℹ  File {i}: No 'massflux' field - will use raw velocity fields directly")
        
        # Setup normalization
        # If stats provided, use them; otherwise compute from this dataset's files
        if normalization_stats is not None:
            self.norm_stats = normalization_stats
            self.norm_helper = NormalizationHelper(normalization_stats, norm_mode=self.norm_mode)
            print("📊 Using provided normalization statistics")
        else:
            self.norm_stats = compute_normalization_stats(
                filenames, start_time, verbose=True
            )
            self.norm_helper = NormalizationHelper(self.norm_stats, norm_mode=self.norm_mode)
        
        # Store key stats as attributes for easy access
        self.temp_min = self.norm_stats['temperature']['min']
        self.temp_max = self.norm_stats['temperature']['max']
        self.temp_range = self.temp_max - self.temp_min
        self.velocity_scale = self.norm_stats['unified_velocity_scale']
        self.sdf_scale = self.norm_stats['sdf']['scale']
        
        # Legacy flag - always normalize now
        self.normalize_temperature = True
        
        print(f"✓ Normalization mode: {self.norm_mode}")
        if self.norm_mode == 'none':
            print(f"   All fields passed through RAW (no normalization)")
        elif self.norm_mode == 'temperature_only':
            print(f"   Temperature: [{self.temp_min:.1f}, {self.temp_max:.1f}]°C → [-1, 1]")
            print(f"   Velocity/SDF: RAW (no normalization)")
        else:
            print(f"   Temperature: [{self.temp_min:.1f}, {self.temp_max:.1f}]°C → [-1, 1]")
            print(f"   Velocity scale: {self.velocity_scale:.4f}")
            print(f"   SDF scale: {self.sdf_scale:.4f}")

    def __len__(self):
        total_len = 0
        for (num_traj, traj_len) in zip(self.num_trajs, self.traj_lens):
            # Ensure we don't go beyond the available timesteps
            available_timesteps = max(0, traj_len - self.start_time)
            total_len += num_traj * available_timesteps
        return total_len

    def _get_interface_velocity(
        self,
        sdf: torch.tensor,
        mass_flux: torch.tensor,
        velx: torch.tensor,
        vely: torch.tensor,
        rho_prime: float,
        dy: float,
        dx: float,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Calculate the interface velocity based on the SDF and mass flux.
        Args:
            sdf: Signed distance function array.
            mass_flux: Mass flux array.
            velx: x-component of velocity.
            vely: y-component of velocity.
            rho_prime: Density of the gas phase divided by density of the liquid phase.
            dy: Grid spacing in the y-direction.
            dx: Grid spacing in the x-direction.
        Returns:
            Tuple of interface velocities (velx_interface, vely_interface).
        """
        # Get interface region which is 1 where mass_flux is non-zero
        interface_region = (mass_flux != 0).float()

        # Get normal vector from SDF and normalize it
        norm_y, norm_x = torch.gradient(sdf, spacing=[dy, dx])
        norm_y = norm_y * interface_region
        norm_x = norm_x * interface_region
        norm_y = norm_y / (torch.norm(torch.stack([norm_x, norm_y]), dim=0) + 1e-8)
        norm_x = norm_x / (torch.norm(torch.stack([norm_x, norm_y]), dim=0) + 1e-8)

        # v_interface = v + (mflx/rho_prime) * norm
        velx_interface = velx * interface_region + (mass_flux / rho_prime) * norm_x
        vely_interface = vely * interface_region + (mass_flux / rho_prime) * norm_y

        return velx_interface, vely_interface

    def _normalize_temperature(self, temperature: torch.Tensor, file_idx: int = None) -> torch.Tensor:
        """Normalize temperature to [-1, 1] range using tanh normalization."""
        return self.norm_helper.normalize_temperature(temperature)
    
    def _denormalize_temperature(self, temperature_norm: torch.Tensor, file_idx: int = None) -> torch.Tensor:
        """Denormalize temperature from [-1, 1] back to original range."""
        return self.norm_helper.denormalize_temperature(temperature_norm)
    
    def _normalize_velocity(self, velocity: torch.Tensor) -> torch.Tensor:
        """Normalize velocity using unified scale (zero-centered)."""
        return self.norm_helper.normalize_velocity(velocity)
    
    def _denormalize_velocity(self, velocity_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize velocity from scaled value back to original."""
        return self.norm_helper.denormalize_velocity(velocity_norm)
    
    def _normalize_sdf(self, sdf: torch.Tensor) -> torch.Tensor:
        """Normalize SDF using zero-preserving scale."""
        return self.norm_helper.normalize_sdf(sdf)
    
    def _denormalize_sdf(self, sdf_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize SDF from scaled value back to original."""
        return self.norm_helper.denormalize_sdf(sdf_norm)
    
    def _normalize_field(self, x: torch.Tensor, field_name: str) -> torch.Tensor:
        """Normalize a field by name."""
        return self.norm_helper.normalize_field(x, field_name)
    
    def _denormalize_field(self, x_norm: torch.Tensor, field_name: str) -> torch.Tensor:
        """Denormalize a field by name."""
        return self.norm_helper.denormalize_field(x_norm, field_name)
    
    def get_temperature_stats(self, file_idx: int = None) -> dict:
        """Get temperature statistics (file_idx kept for backward compatibility)."""
        return {
            'min': self.temp_min,
            'max': self.temp_max,
            'range': self.temp_range
        }
    
    def get_normalization_stats(self) -> Dict[str, Dict[str, float]]:
        """Get all normalization statistics."""
        return self.norm_stats
    
    def get_file_index(self, idx: int) -> int:
        """Get the file index for a given global sample index."""
        samples_per_traj = [
            x * max(0, y - self.start_time)
            for x, y in zip(self.num_trajs, self.traj_lens)
        ]
        cumulative_samples = np.cumsum(samples_per_traj)
        file_idx = np.searchsorted(cumulative_samples, idx, side="right")
        return file_idx
    
    def _downsample(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Downsample a tensor by the configured factor using bilinear interpolation.
        
        Args:
            tensor: Input tensor of shape [C, H, W] or [H, W]
            
        Returns:
            Downsampled tensor
        """
        if self.downsample_factor == 1:
            return tensor
        
        # Handle both [C, H, W] and [H, W] shapes
        if tensor.dim() == 2:
            # Add batch and channel dims: [H, W] -> [1, 1, H, W]
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            downsampled = torch.nn.functional.interpolate(
                tensor, 
                scale_factor=1.0 / self.downsample_factor, 
                mode='bilinear', 
                align_corners=False
            )
            return downsampled.squeeze(0).squeeze(0)
        elif tensor.dim() == 3:
            # Add batch dim: [C, H, W] -> [1, C, H, W]
            tensor = tensor.unsqueeze(0)
            downsampled = torch.nn.functional.interpolate(
                tensor, 
                scale_factor=1.0 / self.downsample_factor, 
                mode='bilinear', 
                align_corners=False
            )
            return downsampled.squeeze(0)
        else:
            return tensor
    
    def _extract_wall_temp_from_filename(self, filename: str) -> float:
        """
        Extract wall temperature from filename.
        Expected format: .../Twall_XX.hdf5 or .../Twall_XX.YY.hdf5
        
        Args:
            filename: Path to HDF5 file
            
        Returns:
            Wall temperature in Celsius as a float
        """
        basename = os.path.basename(filename)
        # Match patterns like Twall_75.hdf5 or Twall_75.5.hdf5
        match = re.search(r'Twall[_-]?(\d+(?:\.\d+)?)', basename, re.IGNORECASE)
        if match:
            return float(match.group(1))
        else:
            # If no match, print warning and return a default value
            print(f"⚠️  Warning: Could not extract wall temperature from filename: {filename}")
            print(f"   Using default wall temperature: 80.0°C")
            return 80.0

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        """
        Get the item at index idx.
        
        The dataset only exposes timesteps >= start_time. For example, if start_time=100:
        - Dataset index 0 maps to actual timestep 100
        - Dataset index 1 maps to actual timestep 101
        - etc.
        
        All fields are normalized according to NORMALIZATION_REQUIREMENTS.md:
        - SDF: Zero-preserving normalization (sdf / scale)
        - Velocity (input): Unified scale normalization (vel / unified_scale)
        - Temperature (output): Tanh normalization to [-1, 1]
        - Velocity (output): Unified scale normalization
        
        Returns:
            Tuple of input and output tensors, and optionally wall temperature.
            If return_wall_temp is False: (inp_data, out_data)
            If return_wall_temp is True: (inp_data, out_data, wall_temp)
        """
        # Calculate how many samples are available per trajectory (after applying start_time)
        samples_per_traj = [
            x * max(0, y - self.start_time)
            for x, y in zip(self.num_trajs, self.traj_lens)
        ]

        cumulative_samples = np.cumsum(samples_per_traj)
        file_idx = np.searchsorted(cumulative_samples, idx, side="right")
        
        # Calculate the local index within the file (0-indexed within available samples)
        local_idx = idx - (cumulative_samples[file_idx - 1] if file_idx > 0 else 0)
        
        # Map local_idx to actual timestep by adding start_time offset
        # This ensures we only access timesteps >= start_time
        timestep = local_idx + self.start_time
        
        # Ensure timestep is within bounds
        max_timestep = self.traj_lens[file_idx] - 1
        if timestep > max_timestep:
            timestep = max_timestep
        
        # Validate that we're respecting start_time
        assert timestep >= self.start_time, f"Timestep {timestep} < start_time {self.start_time}"

        # Form the input tensor
        sdf = torch.tensor(self.data[file_idx]["dfun"][timestep])
        velx = torch.tensor(self.data[file_idx]["velx"][timestep])
        vely = torch.tensor(self.data[file_idx]["vely"][timestep])
        
        # If massflux and fluid params are available, calculate interface velocity
        # Otherwise, use raw velocity fields
        if self.has_interface_velocity[file_idx]:
            mass_flux = torch.tensor(self.data[file_idx]["massflux"][timestep])
            params = self.fluid_params[file_idx]
            rho_prime = params["rhogas"]
            dy = (params["y_max"] - params["y_min"]) / sdf.shape[0]
            dx = (params["x_max"] - params["x_min"]) / sdf.shape[1]

            velx_interface, vely_interface = self._get_interface_velocity(
                sdf, mass_flux, velx, vely, rho_prime, dy, dx
            )
        else:
            # No massflux field - use raw velocity fields directly
            velx_interface = velx
            vely_interface = vely
        
        # Apply downsampling before noise (for fast prototyping)
        sdf = self._downsample(sdf)
        velx_interface = self._downsample(velx_interface)
        vely_interface = self._downsample(vely_interface)
        
        # Apply optical flow noise to conditioning inputs (Task 3)
        # This simulates the uncertainty when SDF and velocity are estimated
        # from optical flow on boiling videos rather than clean simulations
        # Note: Noise is applied BEFORE normalization
        sdf_noisy, velx_noisy, vely_noisy = self.noise_model(
            sdf, velx_interface, vely_interface
        )
        
        # Apply normalization to input fields
        # SDF: Zero-preserving normalization (preserves interface at zero)
        sdf_input = self._normalize_sdf(sdf_noisy)
        # Velocity: Unified scale normalization (preserves direction)
        velx_input = self._normalize_velocity(velx_noisy)
        vely_input = self._normalize_velocity(vely_noisy)
        
        inp_data = torch.stack([sdf_input, velx_input, vely_input])  # (in_C, H, W)

        # Build output tensor with normalization
        out_data = []
        for field in self.output_fields:
            field_data = torch.tensor(self.data[file_idx][field][timestep])
            
            # Apply downsampling
            field_data = self._downsample(field_data)
            
            # Apply normalization based on field type
            field_data = self._normalize_field(field_data, field)
            
            out_data.append(field_data)

        out_data = torch.stack(out_data)  # (out_C, H, W)

        if self.return_wall_temp:
            wall_temp = torch.tensor(self.wall_temps[file_idx], dtype=torch.float32)
            return inp_data.float(), out_data.float(), wall_temp
        else:
            return inp_data.float(), out_data.float()

class BulkFlowAutoregressive(Dataset):
    """
    Dataset class for autoregressive prediction with teacher forcing and scheduled sampling.
    
    Returns consecutive frame pairs where the model conditions on:
    - Current conditioning (SDF, interface velocity)
    - Previous timestep ground truth output (for teacher forcing)
    
    This enables autoregressive training where the model learns to predict
    the current state given the previous state, enforcing temporal consistency
    by construction.
    
    Normalization (based on NORMALIZATION_REQUIREMENTS.md):
    - Temperature: Tanh normalization to [-1, 1]
    - Velocity (velx, vely, velx_interface, vely_interface): Unified shared scale
    - SDF: Zero-preserving normalization (preserves interface at zero)
    
    Training Modes:
    1. Teacher Forcing (scheduled_sampling=False):
        Input: [conditioning_t, output_{t-1}] (ground truth previous state)
        Target: output_t
        Returns: (inp_data_t, prev_output, out_data_t)
    
    2. Scheduled Sampling (scheduled_sampling=True):
        Additional context for model to generate its own prediction:
        Returns: (inp_data_t, prev_output, out_data_t, 
                  conditioning_{t-1}, output_{t-2})
        The model can use conditioning_{t-1} and output_{t-2} to predict 
        output_{t-1} instead of using the ground truth.
    
    Inference (autoregressive rollout):
        Input: [conditioning_t, predicted_{t-1}] (model's own prediction)
        Target: output_t
    
    Args:
        filenames: List of HDF5 file paths
        output_fields: List of output field names (e.g., ["temperature", "velx", "vely"])
        start_time: Starting timestep (to skip initial transients)
        normalization_stats: Pre-computed normalization statistics
        return_wall_temp: Whether to return wall temperature
        noise_cfg: Noise configuration for optical flow simulation
        downsample_factor: Factor to downsample spatial resolution
        scheduled_sampling: Whether to return extra context for scheduled sampling
        normalize_temperature: Legacy parameter, ignored
    """

    def __init__(
        self,
        filenames: List[str],
        output_fields: Optional[List[str]] = None,
        start_time: int = 0,
        normalization_stats: Optional[Dict[str, Dict[str, float]]] = None,
        return_wall_temp: bool = False,
        noise_cfg: Optional[Dict[str, Any]] = None,
        downsample_factor: int = 1,
        scheduled_sampling: bool = False,
        norm_mode: str = 'all',
        # Legacy parameter - kept for backward compatibility but ignored
        normalize_temperature: bool = True,
    ):
        super().__init__()
        self.filenames = filenames
        self.return_wall_temp = return_wall_temp
        self.downsample_factor = downsample_factor
        self.scheduled_sampling = scheduled_sampling
        self.norm_mode = norm_mode
        
        # Initialize noise model
        self.noise_model = create_noise_model(noise_cfg)
        if noise_cfg is not None and noise_cfg.get('enabled', True):
            noise_type = noise_cfg.get('noise_type', 'optical_flow')
            print(f"\n🔊 Noise Model Enabled (type: {noise_type}):")
            print(f"   {self.noise_model}")

        if output_fields is not None:
            self.output_fields = output_fields
        else:
            self.output_fields = ["temperature", "velx", "vely"]
        
        # Log configuration
        if self.downsample_factor > 1:
            print(f"\n📐 Downsampling Enabled: {self.downsample_factor}x")
        
        if self.scheduled_sampling:
            print(f"\n🔄 Autoregressive Dataset (Scheduled Sampling Mode):")
            print(f"   Input: [conditioning_t, output_(t-1)]")
            print(f"   Target: output_t")
            print(f"   Extra context: [conditioning_(t-1), output_(t-2)]")
            print(f"   ℹ️  Model decides whether to use GT or predictions")
        else:
            print(f"\n🔄 Autoregressive Dataset (Teacher Forcing):")
            print(f"   Input: [conditioning_t, output_(t-1)]")
            print(f"   Target: output_t")

        # Effective start time depends on mode:
        # - Teacher forcing: need t-1, so start at 1
        # - Scheduled sampling: need t-2, so start at 2
        self.base_start_time = start_time
        if self.scheduled_sampling:
            self.effective_start_time = max(start_time, 2)  # Need t-2 for scheduled sampling
        else:
            self.effective_start_time = max(start_time, 1)  # Need t-1 for teacher forcing
        
        if self.effective_start_time > start_time:
            mode_str = "scheduled sampling" if self.scheduled_sampling else "autoregressive"
            print(f"   ⚠️ start_time adjusted from {start_time} to {self.effective_start_time} for {mode_str}")

        self.data = [h5.File(filename, "r") for filename in filenames]
        self.num_trajs = []
        self.traj_lens = []

        for h5_file in self.data:
            self.num_trajs.append(1)
            total_timesteps = h5_file["dfun"].shape[0]
            self.traj_lens.append(total_timesteps)
        
        # Calculate available samples
        total_available = sum(max(0, traj_len - self.effective_start_time) for traj_len in self.traj_lens)
        print(f"   Available samples: {total_available}")
        
        # Extract wall temperatures
        self.wall_temps = []
        for filename in filenames:
            wall_temp = self._extract_wall_temp_from_filename(filename)
            self.wall_temps.append(wall_temp)

        # Load fluid params and check for massflux field
        self.fluid_params_files = [fname.replace(".hdf5", ".json") for fname in filenames]
        self.fluid_params = []
        self.has_interface_velocity = []
        
        for i, (h5_file, fluid_params_file) in enumerate(zip(self.data, self.fluid_params_files)):
            has_massflux = "massflux" in h5_file.keys()
            
            fluid_params = None
            if has_massflux:
                try:
                    with open(fluid_params_file, "r", encoding="utf-8") as f:
                        fluid_params = json.load(f)
                except FileNotFoundError:
                    pass
            
            self.fluid_params.append(fluid_params)
            can_calc_interface = has_massflux and (fluid_params is not None)
            self.has_interface_velocity.append(can_calc_interface)
        
        # Setup normalization
        if normalization_stats is not None:
            self.norm_stats = normalization_stats
            self.norm_helper = NormalizationHelper(normalization_stats, norm_mode=self.norm_mode)
            print(f"   📊 Using provided normalization statistics (mode: {self.norm_mode})")
        else:
            self.norm_stats = compute_normalization_stats(
                filenames, start_time, verbose=True
            )
            self.norm_helper = NormalizationHelper(self.norm_stats, norm_mode=self.norm_mode)
        
        # Store key stats as attributes for easy access
        self.temp_min = self.norm_stats['temperature']['min']
        self.temp_max = self.norm_stats['temperature']['max']
        self.temp_range = self.temp_max - self.temp_min
        self.velocity_scale = self.norm_stats['unified_velocity_scale']
        self.sdf_scale = self.norm_stats['sdf']['scale']
        
        # Legacy flag - always normalize now
        self.normalize_temperature = True
        
        print(f"   ✓ Normalization: T=[{self.temp_min:.1f},{self.temp_max:.1f}]°C, V_scale={self.velocity_scale:.4f}, SDF_scale={self.sdf_scale:.4f}")

    def __len__(self):
        total_len = 0
        for (num_traj, traj_len) in zip(self.num_trajs, self.traj_lens):
            # Each sample needs current timestep t and previous timestep t-1
            available = max(0, traj_len - self.effective_start_time)
            total_len += num_traj * available
        return total_len

    def _get_interface_velocity(
        self,
        sdf: torch.tensor,
        mass_flux: torch.tensor,
        velx: torch.tensor,
        vely: torch.tensor,
        rho_prime: float,
        dy: float,
        dx: float,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Calculate interface velocity."""
        interface_region = (mass_flux != 0).float()
        norm_y, norm_x = torch.gradient(sdf, spacing=[dy, dx])
        norm_y = norm_y * interface_region
        norm_x = norm_x * interface_region
        norm_y = norm_y / (torch.norm(torch.stack([norm_x, norm_y]), dim=0) + 1e-8)
        norm_x = norm_x / (torch.norm(torch.stack([norm_x, norm_y]), dim=0) + 1e-8)
        velx_interface = velx * interface_region + (mass_flux / rho_prime) * norm_x
        vely_interface = vely * interface_region + (mass_flux / rho_prime) * norm_y
        return velx_interface, vely_interface

    def _normalize_temperature(self, temperature: torch.Tensor, file_idx: int = None) -> torch.Tensor:
        """Normalize temperature to [-1, 1] range using tanh normalization."""
        return self.norm_helper.normalize_temperature(temperature)
    
    def _denormalize_temperature(self, temperature_norm: torch.Tensor, file_idx: int = None) -> torch.Tensor:
        """Denormalize temperature from [-1, 1] back to original range."""
        return self.norm_helper.denormalize_temperature(temperature_norm)
    
    def _normalize_velocity(self, velocity: torch.Tensor) -> torch.Tensor:
        """Normalize velocity using unified scale."""
        return self.norm_helper.normalize_velocity(velocity)
    
    def _denormalize_velocity(self, velocity_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize velocity from scaled value back to original."""
        return self.norm_helper.denormalize_velocity(velocity_norm)
    
    def _normalize_sdf(self, sdf: torch.Tensor) -> torch.Tensor:
        """Normalize SDF using zero-preserving scale."""
        return self.norm_helper.normalize_sdf(sdf)
    
    def _denormalize_sdf(self, sdf_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize SDF from scaled value back to original."""
        return self.norm_helper.denormalize_sdf(sdf_norm)
    
    def _normalize_field(self, x: torch.Tensor, field_name: str) -> torch.Tensor:
        """Normalize a field by name."""
        return self.norm_helper.normalize_field(x, field_name)
    
    def _denormalize_field(self, x_norm: torch.Tensor, field_name: str) -> torch.Tensor:
        """Denormalize a field by name."""
        return self.norm_helper.denormalize_field(x_norm, field_name)
    
    def get_normalization_stats(self) -> Dict[str, Dict[str, float]]:
        """Get all normalization statistics."""
        return self.norm_stats
    
    def _extract_wall_temp_from_filename(self, filename: str) -> float:
        """Extract wall temperature from filename."""
        basename = os.path.basename(filename)
        match = re.search(r'Twall[_-]?(\d+(?:\.\d+)?)', basename, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 80.0
    
    def _downsample(self, tensor: torch.Tensor) -> torch.Tensor:
        """Downsample tensor by configured factor."""
        if self.downsample_factor == 1:
            return tensor
        
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            downsampled = torch.nn.functional.interpolate(
                tensor, scale_factor=1.0 / self.downsample_factor, 
                mode='bilinear', align_corners=False
            )
            return downsampled.squeeze(0).squeeze(0)
        elif tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            downsampled = torch.nn.functional.interpolate(
                tensor, scale_factor=1.0 / self.downsample_factor, 
                mode='bilinear', align_corners=False
            )
            return downsampled.squeeze(0)
        return tensor

    def _get_conditioning(self, file_idx: int, timestep: int) -> torch.Tensor:
        """
        Get conditioning inputs (SDF, interface velocity) for a single timestep.
        
        All fields are normalized according to NORMALIZATION_REQUIREMENTS.md.
        
        Returns:
            inp_data: [C_in, H, W] - Normalized SDF, velx_interface, vely_interface
        """
        sdf = torch.tensor(self.data[file_idx]["dfun"][timestep])
        velx = torch.tensor(self.data[file_idx]["velx"][timestep])
        vely = torch.tensor(self.data[file_idx]["vely"][timestep])
        
        if self.has_interface_velocity[file_idx]:
            mass_flux = torch.tensor(self.data[file_idx]["massflux"][timestep])
            params = self.fluid_params[file_idx]
            rho_prime = params["rhogas"]
            dy = (params["y_max"] - params["y_min"]) / sdf.shape[0]
            dx = (params["x_max"] - params["x_min"]) / sdf.shape[1]
            velx_interface, vely_interface = self._get_interface_velocity(
                sdf, mass_flux, velx, vely, rho_prime, dy, dx
            )
        else:
            velx_interface = velx
            vely_interface = vely
        
        # Downsample
        sdf = self._downsample(sdf)
        velx_interface = self._downsample(velx_interface)
        vely_interface = self._downsample(vely_interface)
        
        # Apply noise to conditioning inputs (before normalization)
        sdf_noisy, velx_noisy, vely_noisy = self.noise_model(
            sdf, velx_interface, vely_interface
        )
        
        # Normalize input fields
        sdf_input = self._normalize_sdf(sdf_noisy)
        velx_input = self._normalize_velocity(velx_noisy)
        vely_input = self._normalize_velocity(vely_noisy)
        
        inp_data = torch.stack([sdf_input, velx_input, vely_input])
        return inp_data

    def _get_output(self, file_idx: int, timestep: int) -> torch.Tensor:
        """
        Get output fields for a single timestep.
        
        All fields are normalized according to NORMALIZATION_REQUIREMENTS.md.
        
        Returns:
            out_data: [C_out, H, W] - Normalized output fields (e.g., temperature, velx, vely)
        """
        out_data = []
        for field in self.output_fields:
            field_data = torch.tensor(self.data[file_idx][field][timestep])
            field_data = self._downsample(field_data)
            # Normalize based on field type
            field_data = self._normalize_field(field_data, field)
            out_data.append(field_data)
        return torch.stack(out_data)

    def __getitem__(self, idx: int):
        """
        Get a sample for autoregressive training.
        
        Teacher Forcing Mode (scheduled_sampling=False):
            Returns:
                inp_data_t: [C_in, H, W] - Current conditioning (SDF, interface vel)
                prev_output: [C_out, H, W] - Previous timestep output (ground truth)
                out_data_t: [C_out, H, W] - Current timestep target
                wall_temp: (optional) Wall temperature
        
        Scheduled Sampling Mode (scheduled_sampling=True):
            Returns:
                inp_data_t: [C_in, H, W] - Current conditioning at t
                prev_output: [C_out, H, W] - Ground truth output at t-1
                out_data_t: [C_out, H, W] - Target output at t
                conditioning_t_minus_1: [C_in, H, W] - Conditioning at t-1
                output_t_minus_2: [C_out, H, W] - Ground truth output at t-2
                wall_temp: (optional) Wall temperature
        """
        # Calculate available samples per trajectory
        samples_per_traj = [
            x * max(0, y - self.effective_start_time)
            for x, y in zip(self.num_trajs, self.traj_lens)
        ]

        cumulative_samples = np.cumsum(samples_per_traj)
        file_idx = np.searchsorted(cumulative_samples, idx, side="right")
        
        local_idx = idx - (cumulative_samples[file_idx - 1] if file_idx > 0 else 0)
        
        # Current timestep t (starts at effective_start_time)
        timestep_t = local_idx + self.effective_start_time
        
        # Previous timesteps
        timestep_t_minus_1 = timestep_t - 1
        timestep_t_minus_2 = timestep_t - 2
        
        # Ensure within bounds
        max_timestep = self.traj_lens[file_idx] - 1
        timestep_t = min(timestep_t, max_timestep)
        timestep_t_minus_1 = max(0, timestep_t_minus_1)
        timestep_t_minus_2 = max(0, timestep_t_minus_2)
        
        # Get current conditioning
        inp_data_t = self._get_conditioning(file_idx, timestep_t)
        
        # Get previous output (for teacher forcing)
        prev_output = self._get_output(file_idx, timestep_t_minus_1)
        
        # Get current output (target)
        out_data_t = self._get_output(file_idx, timestep_t)

        if self.scheduled_sampling:
            # Get extra context for scheduled sampling
            # conditioning at t-1 (to generate model's prediction of output at t-1)
            conditioning_t_minus_1 = self._get_conditioning(file_idx, timestep_t_minus_1)
            
            # output at t-2 (to condition the prediction of output at t-1)
            output_t_minus_2 = self._get_output(file_idx, timestep_t_minus_2)
            
            if self.return_wall_temp:
                wall_temp = torch.tensor(self.wall_temps[file_idx], dtype=torch.float32)
                return (inp_data_t.float(), prev_output.float(), out_data_t.float(),
                        conditioning_t_minus_1.float(), output_t_minus_2.float(), wall_temp)
            else:
                return (inp_data_t.float(), prev_output.float(), out_data_t.float(),
                        conditioning_t_minus_1.float(), output_t_minus_2.float())
        else:
            # Standard teacher forcing mode
            if self.return_wall_temp:
                wall_temp = torch.tensor(self.wall_temps[file_idx], dtype=torch.float32)
                return inp_data_t.float(), prev_output.float(), out_data_t.float(), wall_temp
            else:
                return inp_data_t.float(), prev_output.float(), out_data_t.float()

class BulkFlowARBootstrap(Dataset):
    """
    Dataset class for autoregressive training with bootstrap initialization.
    
    This dataset returns trajectory segments with conditioning history for 
    bootstrap training. The model learns to:
    1. Infer initial bulk state from conditioning history (bootstrap mode)
    2. Predict state transitions given previous state (AR mode)
    
    Returns for each sample:
    - conditioning_history: [T_hist, C_cond, H, W] - history before the segment
    - conditioning_sequence: [L, C_cond, H, W] - conditioning for the rollout segment
    - target_sequence: [L, C_out, H, W] - ground truth targets for the segment
    
    The first frame of the segment uses bootstrap mode (infer from history),
    subsequent frames use AR mode (condition on previous state).
    
    Args:
        filenames: List of HDF5 file paths
        output_fields: List of output field names (e.g., ["temperature", "velx", "vely"])
        start_time: Starting timestep (to skip initial transients)
        history_length: Number of frames in conditioning history for bootstrap
        history_stride: Stride between history frames (1 = consecutive, 2 = every other, etc.)
        rollout_length: Number of timesteps in each training segment
        normalization_stats: Pre-computed normalization statistics
        return_wall_temp: Whether to return wall temperature
        noise_cfg: Noise configuration for optical flow simulation
        downsample_factor: Factor to downsample spatial resolution
    """
    
    def __init__(
        self,
        filenames: List[str],
        output_fields: Optional[List[str]] = None,
        start_time: int = 0,
        history_length: int = 10,
        history_stride: int = 1,
        rollout_length: int = 5,
        normalization_stats: Optional[Dict[str, Dict[str, float]]] = None,
        return_wall_temp: bool = False,
        noise_cfg: Optional[Dict[str, Any]] = None,
        downsample_factor: int = 1,
        norm_mode: str = 'all',
    ):
        super().__init__()
        self.filenames = filenames
        self.return_wall_temp = return_wall_temp
        self.history_length = history_length
        self.history_stride = history_stride
        self.rollout_length = rollout_length
        self.downsample_factor = downsample_factor
        self.norm_mode = norm_mode
        
        # Initialize noise model
        self.noise_model = create_noise_model(noise_cfg)
        if noise_cfg is not None and noise_cfg.get('enabled', True):
            noise_type = noise_cfg.get('noise_type', 'optical_flow')
            print(f"\n🔊 Noise Model Enabled (type: {noise_type}):")
            print(f"   {self.noise_model}")
        
        if output_fields is not None:
            self.output_fields = output_fields
        else:
            self.output_fields = ["temperature", "velx", "vely"]
        
        # Total temporal context required (history spans history_length * history_stride timesteps)
        self.history_span = history_length * history_stride
        self.total_context = self.history_span + rollout_length
        
        print(f"\n🚀 Bootstrap AR Dataset Configuration:")
        print(f"   History length: {self.history_length} frames")
        print(f"   History stride: {self.history_stride} (spans {self.history_span} timesteps)")
        print(f"   Rollout length: {self.rollout_length}")
        print(f"   Total context: {self.total_context} timesteps")
        
        # Effective start time must accommodate the full history span
        self.base_start_time = start_time
        self.effective_start_time = max(start_time, self.history_span)
        
        if self.effective_start_time > start_time:
            print(f"   ⚠️ start_time adjusted from {start_time} to {self.effective_start_time}")
        
        self.data = [h5.File(filename, "r") for filename in filenames]
        self.num_trajs = []
        self.traj_lens = []
        
        for h5_file in self.data:
            self.num_trajs.append(1)
            total_timesteps = h5_file["dfun"].shape[0]
            self.traj_lens.append(total_timesteps)
        
        # Calculate available samples (need history + rollout)
        total_available = sum(
            max(0, traj_len - self.effective_start_time - self.rollout_length + 1)
            for traj_len in self.traj_lens
        )
        print(f"   Available samples: {total_available}")
        
        # Extract wall temperatures
        self.wall_temps = []
        for filename in filenames:
            wall_temp = self._extract_wall_temp_from_filename(filename)
            self.wall_temps.append(wall_temp)
        
        # Load fluid params and check for massflux field
        self.fluid_params_files = [fname.replace(".hdf5", ".json") for fname in filenames]
        self.fluid_params = []
        self.has_interface_velocity = []
        
        for i, (h5_file, fluid_params_file) in enumerate(zip(self.data, self.fluid_params_files)):
            has_massflux = "massflux" in h5_file.keys()
            
            fluid_params = None
            if has_massflux:
                try:
                    with open(fluid_params_file, "r", encoding="utf-8") as f:
                        fluid_params = json.load(f)
                except FileNotFoundError:
                    pass
            
            self.fluid_params.append(fluid_params)
            can_calc_interface = has_massflux and (fluid_params is not None)
            self.has_interface_velocity.append(can_calc_interface)
        
        # Setup normalization
        if normalization_stats is not None:
            self.norm_stats = normalization_stats
            self.norm_helper = NormalizationHelper(normalization_stats)
            print("   📊 Using provided normalization statistics")
        else:
            self.norm_stats = compute_normalization_stats(
                filenames, start_time, verbose=True
            )
            self.norm_helper = NormalizationHelper(self.norm_stats)
        
        # Store key stats
        self.temp_min = self.norm_stats['temperature']['min']
        self.temp_max = self.norm_stats['temperature']['max']
        self.temp_range = self.temp_max - self.temp_min
        self.velocity_scale = self.norm_stats['unified_velocity_scale']
        self.sdf_scale = self.norm_stats['sdf']['scale']
        
        print(f"   ✓ Normalization: T=[{self.temp_min:.1f},{self.temp_max:.1f}]°C")
    
    def __len__(self):
        total_len = 0
        for (num_traj, traj_len) in zip(self.num_trajs, self.traj_lens):
            # Each sample needs history + rollout
            available = max(0, traj_len - self.effective_start_time - self.rollout_length + 1)
            total_len += num_traj * available
        return total_len
    
    def _get_interface_velocity(
        self,
        sdf: torch.tensor,
        mass_flux: torch.tensor,
        velx: torch.tensor,
        vely: torch.tensor,
        rho_prime: float,
        dy: float,
        dx: float,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Calculate interface velocity."""
        interface_region = (mass_flux != 0).float()
        norm_y, norm_x = torch.gradient(sdf, spacing=[dy, dx])
        norm_y = norm_y * interface_region
        norm_x = norm_x * interface_region
        norm_y = norm_y / (torch.norm(torch.stack([norm_x, norm_y]), dim=0) + 1e-8)
        norm_x = norm_x / (torch.norm(torch.stack([norm_x, norm_y]), dim=0) + 1e-8)
        velx_interface = velx * interface_region + (mass_flux / rho_prime) * norm_x
        vely_interface = vely * interface_region + (mass_flux / rho_prime) * norm_y
        return velx_interface, vely_interface
    
    def _normalize_field(self, x: torch.Tensor, field_name: str) -> torch.Tensor:
        """Normalize a field by name."""
        return self.norm_helper.normalize_field(x, field_name)
    
    def _denormalize_field(self, x_norm: torch.Tensor, field_name: str) -> torch.Tensor:
        """Denormalize a field by name."""
        return self.norm_helper.denormalize_field(x_norm, field_name)
    
    def _extract_wall_temp_from_filename(self, filename: str) -> float:
        """Extract wall temperature from filename."""
        basename = os.path.basename(filename)
        match = re.search(r'Twall[_-]?(\d+(?:\.\d+)?)', basename, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 80.0
    
    def _downsample(self, tensor: torch.Tensor) -> torch.Tensor:
        """Downsample tensor by configured factor."""
        if self.downsample_factor == 1:
            return tensor
        
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            downsampled = torch.nn.functional.interpolate(
                tensor, scale_factor=1.0 / self.downsample_factor,
                mode='bilinear', align_corners=False
            )
            return downsampled.squeeze(0).squeeze(0)
        elif tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            downsampled = torch.nn.functional.interpolate(
                tensor, scale_factor=1.0 / self.downsample_factor,
                mode='bilinear', align_corners=False
            )
            return downsampled.squeeze(0)
        return tensor
    
    def _get_conditioning_frame(self, file_idx: int, timestep: int) -> torch.Tensor:
        """
        Get conditioning inputs (SDF, interface velocity) for a single timestep.
        
        Returns:
            [C_cond, H, W] - Normalized conditioning
        """
        sdf = torch.tensor(self.data[file_idx]["dfun"][timestep])
        velx = torch.tensor(self.data[file_idx]["velx"][timestep])
        vely = torch.tensor(self.data[file_idx]["vely"][timestep])
        
        if self.has_interface_velocity[file_idx]:
            mass_flux = torch.tensor(self.data[file_idx]["massflux"][timestep])
            params = self.fluid_params[file_idx]
            rho_prime = params["rhogas"]
            dy = (params["y_max"] - params["y_min"]) / sdf.shape[0]
            dx = (params["x_max"] - params["x_min"]) / sdf.shape[1]
            velx_interface, vely_interface = self._get_interface_velocity(
                sdf, mass_flux, velx, vely, rho_prime, dy, dx
            )
        else:
            velx_interface = velx
            vely_interface = vely
        
        # Downsample
        sdf = self._downsample(sdf)
        velx_interface = self._downsample(velx_interface)
        vely_interface = self._downsample(vely_interface)
        
        # Apply noise
        sdf_noisy, velx_noisy, vely_noisy = self.noise_model(
            sdf, velx_interface, vely_interface
        )
        
        # Normalize
        sdf_norm = self.norm_helper.normalize_sdf(sdf_noisy)
        velx_norm = self.norm_helper.normalize_velocity(velx_noisy)
        vely_norm = self.norm_helper.normalize_velocity(vely_noisy)
        
        return torch.stack([sdf_norm, velx_norm, vely_norm])
    
    def _get_output_frame(self, file_idx: int, timestep: int) -> torch.Tensor:
        """
        Get output fields for a single timestep.
        
        Returns:
            [C_out, H, W] - Normalized outputs
        """
        out_data = []
        for field in self.output_fields:
            field_data = torch.tensor(self.data[file_idx][field][timestep])
            field_data = self._downsample(field_data)
            field_data = self._normalize_field(field_data, field)
            out_data.append(field_data)
        return torch.stack(out_data)
    
    def get_normalization_stats(self) -> Dict[str, Dict[str, float]]:
        """Get all normalization statistics."""
        return self.norm_stats
    
    def __getitem__(self, idx: int):
        """
        Get a trajectory segment with conditioning history.
        
        Returns:
            conditioning_history: [T_hist, C_cond, H, W] - for bootstrap
            conditioning_sequence: [L, C_cond, H, W] - rollout conditioning
            target_sequence: [L, C_out, H, W] - rollout targets
            wall_temp: (optional) Wall temperature
        """
        # Calculate available samples per trajectory
        samples_per_traj = [
            x * max(0, y - self.effective_start_time - self.rollout_length + 1)
            for x, y in zip(self.num_trajs, self.traj_lens)
        ]
        
        cumulative_samples = np.cumsum(samples_per_traj)
        file_idx = np.searchsorted(cumulative_samples, idx, side="right")
        
        local_idx = idx - (cumulative_samples[file_idx - 1] if file_idx > 0 else 0)
        
        # Segment starts at this timestep (first frame of rollout)
        segment_start = local_idx + self.effective_start_time
        
        # Ensure within bounds
        max_start = self.traj_lens[file_idx] - self.rollout_length
        segment_start = min(segment_start, max_start)
        
        # Collect conditioning history (before segment, sampled with stride)
        history_start = segment_start - self.history_length * self.history_stride
        conditioning_history = []
        for t in range(history_start, segment_start, self.history_stride):
            t_clamped = max(0, t)  # Handle edge case
            cond_frame = self._get_conditioning_frame(file_idx, t_clamped)
            conditioning_history.append(cond_frame)
        conditioning_history = torch.stack(conditioning_history)  # [T_hist, C_cond, H, W]
        
        # Collect rollout segment
        conditioning_sequence = []
        target_sequence = []
        for t in range(segment_start, segment_start + self.rollout_length):
            cond_frame = self._get_conditioning_frame(file_idx, t)
            out_frame = self._get_output_frame(file_idx, t)
            conditioning_sequence.append(cond_frame)
            target_sequence.append(out_frame)
        
        conditioning_sequence = torch.stack(conditioning_sequence)  # [L, C_cond, H, W]
        target_sequence = torch.stack(target_sequence)  # [L, C_out, H, W]
        
        if self.return_wall_temp:
            wall_temp = torch.tensor(self.wall_temps[file_idx], dtype=torch.float32)
            return (conditioning_history.float(), conditioning_sequence.float(),
                    target_sequence.float(), wall_temp)
        else:
            return (conditioning_history.float(), conditioning_sequence.float(),
                    target_sequence.float())


class BulkFlowHistory(BulkFlow):
    """
    Dataset for history-window flow matching (non-autoregressive baseline).

    Returns a sliding window of W conditioning frames flattened into channels,
    plus the target at the last timestep.  The tuple format matches BulkFlow:
        (inp_data, out_data)
    where inp_data has shape [W * C_cond, H, W] and out_data is [C_out, H, W].

    With ``history_stride = S`` the window covers the strided indices
    ``[t - (W-1)*S, t - (W-2)*S, ..., t - S, t]``, i.e. it spans
    ``(W-1)*S + 1`` raw timesteps while keeping the same channel count
    (and therefore the same compute) as ``S = 1``.

    Args:
        history_window: Number of conditioning frames in the sliding window (W).
        history_stride: Stride between conditioning frames (1 = consecutive,
            2 = every other frame, etc.). Defaults to 1 for backward
            compatibility with existing checkpoints.
        All other args are forwarded to BulkFlow.
    """

    def __init__(
        self,
        filenames: List[str],
        output_fields: Optional[List[str]] = None,
        start_time: int = 0,
        history_window: int = 10,
        history_stride: int = 1,
        normalization_stats: Optional[Dict[str, Dict[str, float]]] = None,
        return_wall_temp: bool = False,
        noise_cfg: Optional[Dict[str, Any]] = None,
        downsample_factor: int = 1,
        norm_mode: str = 'all',
    ):
        if history_stride < 1:
            raise ValueError(f"history_stride must be >= 1, got {history_stride}")

        self.history_window = history_window
        self.history_stride = history_stride
        self.history_span = (history_window - 1) * history_stride + 1
        self._effective_start_time = max(start_time, self.history_span - 1)

        super().__init__(
            filenames=filenames,
            output_fields=output_fields,
            start_time=self._effective_start_time,
            normalization_stats=normalization_stats,
            return_wall_temp=return_wall_temp,
            noise_cfg=noise_cfg,
            downsample_factor=downsample_factor,
            norm_mode=norm_mode,
        )

        print(f"\n   History-Window Dataset Configuration:")
        print(f"   History window (W): {self.history_window}")
        print(f"   History stride (S): {self.history_stride}")
        print(f"   Temporal span:      {self.history_span} raw frames "
              f"(= (W-1)*S + 1)")
        print(f"   Effective start time: {self._effective_start_time}")
        print(f"   Samples: {len(self)}")

    def _get_conditioning_frame(self, file_idx: int, timestep: int) -> torch.Tensor:
        """
        Get normalized conditioning [C_cond, H, W] for a single timestep.

        Mirrors BulkFlowARBootstrap._get_conditioning_frame.
        """
        sdf = torch.tensor(self.data[file_idx]["dfun"][timestep])
        velx = torch.tensor(self.data[file_idx]["velx"][timestep])
        vely = torch.tensor(self.data[file_idx]["vely"][timestep])

        if self.has_interface_velocity[file_idx]:
            mass_flux = torch.tensor(self.data[file_idx]["massflux"][timestep])
            params = self.fluid_params[file_idx]
            rho_prime = params["rhogas"]
            dy = (params["y_max"] - params["y_min"]) / sdf.shape[0]
            dx = (params["x_max"] - params["x_min"]) / sdf.shape[1]
            velx_interface, vely_interface = self._get_interface_velocity(
                sdf, mass_flux, velx, vely, rho_prime, dy, dx
            )
        else:
            velx_interface = velx
            vely_interface = vely

        sdf = self._downsample(sdf)
        velx_interface = self._downsample(velx_interface)
        vely_interface = self._downsample(vely_interface)

        sdf_noisy, velx_noisy, vely_noisy = self.noise_model(
            sdf, velx_interface, vely_interface
        )

        sdf_norm = self._normalize_sdf(sdf_noisy)
        velx_norm = self._normalize_velocity(velx_noisy)
        vely_norm = self._normalize_velocity(vely_noisy)

        return torch.stack([sdf_norm, velx_norm, vely_norm])

    def __getitem__(self, idx: int):
        """
        Returns (conditioning_window_flat, out_data) and optionally wall_temp.

        conditioning_window_flat: [W * C_cond, H, W]
        out_data:                 [C_out, H, W]
        """
        samples_per_traj = [
            x * max(0, y - self.start_time)
            for x, y in zip(self.num_trajs, self.traj_lens)
        ]

        cumulative_samples = np.cumsum(samples_per_traj)
        file_idx = np.searchsorted(cumulative_samples, idx, side="right")

        local_idx = idx - (cumulative_samples[file_idx - 1] if file_idx > 0 else 0)
        timestep = local_idx + self.start_time

        max_timestep = self.traj_lens[file_idx] - 1
        timestep = min(timestep, max_timestep)

        # Collect W conditioning frames with stride S:
        #   [t - (W-1)*S, t - (W-2)*S, ..., t - S, t]
        # Frames before the start of the trajectory are clamped to t=0
        # (same boundary policy as the stride=1 case).
        cond_frames = []
        S = self.history_stride
        oldest = timestep - (self.history_window - 1) * S
        for t in range(oldest, timestep + 1, S):
            t_clamped = max(0, t)
            cond_frames.append(self._get_conditioning_frame(file_idx, t_clamped))

        # Flatten W frames into channels: list of [C_cond, H, W] -> [W*C_cond, H, W]
        conditioning_window_flat = torch.cat(cond_frames, dim=0)

        # Build output at timestep t (same logic as BulkFlow)
        out_data = []
        for field in self.output_fields:
            field_data = torch.tensor(self.data[file_idx][field][timestep])
            field_data = self._downsample(field_data)
            field_data = self._normalize_field(field_data, field)
            out_data.append(field_data)
        out_data = torch.stack(out_data)

        if self.return_wall_temp:
            wall_temp = torch.tensor(self.wall_temps[file_idx], dtype=torch.float32)
            return conditioning_window_flat.float(), out_data.float(), wall_temp
        return conditioning_window_flat.float(), out_data.float()
