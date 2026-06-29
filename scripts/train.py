import os
import pprint
import time
import signal
import json

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from torch.utils.data import DataLoader
from lightning import seed_everything, Trainer
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelSummary, Callback, ModelCheckpoint
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.loggers import WandbLogger

from bubblefusion.data import BulkFlow, BulkFlowAutoregressive, BulkFlowHistory
from bubblefusion.data.bubbleml import compute_normalization_stats, BulkFlowARBootstrap
from bubblefusion.models import BubbleDDPMLightning, UNetLightning, UNetARLightning, ConditionalFlowMatchingLightning, ConditionalFlowMatchingJiTLightning, ScoreBasedVESDELightning, ConditionalFlowMatchingARLightning, ConditionalFlowMatchingARBootstrapLightning, FFNOLightning, EDMLightning, EDMARBootstrapLightning, DiffusionPDELightning, ConditionalFlowMatchingHistoryLightning

def is_leader_process():
    """
    Check if the current process is the leader process.
    """
    if os.getenv("SLURM_PROCID") is None:
        if os.getenv("LOCAL_RANK") is not None:
            return int(os.getenv("LOCAL_RANK")) == 0
        else:
            return True
    else:
        return os.getenv("SLURM_PROCID") == "0"

def find_latest_checkpoint(log_dir):
    """
    Find the latest checkpoint in the given log directory.
    Looks for both Lightning checkpoints and HPC preemption checkpoints.
    """
    import glob
    
    checkpoint_patterns = [
        os.path.join(log_dir, "lightning_logs", "**", "checkpoints", "*.ckpt"),
        os.path.join(log_dir, "checkpoints", "*.ckpt"),
        os.path.join(log_dir, "hpc_ckpt_*.ckpt"),
        os.path.join(log_dir, "*.ckpt")
    ]
    
    latest_checkpoint = None
    latest_time = 0
    
    for pattern in checkpoint_patterns:
        checkpoints = glob.glob(pattern, recursive=True)
        for ckpt in checkpoints:
            if os.path.isfile(ckpt):
                mtime = os.path.getmtime(ckpt)
                if mtime > latest_time:
                    latest_time = mtime
                    latest_checkpoint = ckpt
    
    return latest_checkpoint

class PreemptionCheckpointCallback(Callback):
    """
    Tries to save a checkpoint when a SIGTERM signal is received.
    Args:
        checkpoint_path: Path to save the checkpoint.
    """
    def __init__(self, checkpoint_path="preemption_checkpoint.ckpt"):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.already_handled = False

    def setup(self, trainer, pl_module, stage: str) -> None:
        self.trainer = trainer
        # Register the signal handler for SIGTERM in case of job preemption due to paid job
        signal.signal(signal.SIGTERM, self.handle_preemption)

    def handle_preemption(self, signum, frame):
        """
        Handle the SIGTERM signal.
        """
        if self.already_handled:
            return
        self.already_handled = True
        try:
            # Save the checkpoint. Use trainer.save_checkpoint if accessible.
            # Note: You might need to call this on the main thread.
            self.trainer.save_checkpoint(self.checkpoint_path)
            print(f"Due to preemption Checkpoint saved to {self.checkpoint_path}.")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
        # Optionally, delay a bit to ensure the checkpoint save finishes.
        time.sleep(5)

@hydra.main(version_base=None, config_path="../bubblefusion/config", config_name="default")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    params = {}
    params["nodes"] = cfg.nodes
    params["devices"] = cfg.devices
    params["checkpoint_path"] = cfg.checkpoint_path
    params["data_cfg"] = cfg.data_cfg
    params["model_cfg"] = cfg.model_cfg
    params["optim_cfg"] =  cfg.optim_cfg
    params["scheduler_cfg"] =  cfg.scheduler_cfg
    params["task_cfg"] = getattr(cfg, 'task_cfg', None)
    params["use_wandb"] = cfg.use_wandb
    params["wandb_cfg"] = cfg.wandb_cfg if hasattr(cfg, 'wandb_cfg') else {}
    params["auto_resume"] = getattr(cfg, 'auto_resume', True)
    params["resume_from_last"] = getattr(cfg, 'resume_from_last', True)
    params["checkpoint_monitor"] = getattr(cfg, 'checkpoint_monitor', 'val_loss')
    params["checkpoint_mode"] = getattr(cfg, 'checkpoint_mode', 'min')
    params["save_top_k"] = getattr(cfg, 'save_top_k', 3)
    params["save_last"] = getattr(cfg, 'save_last', True)

    # Determine log directory and checkpoint paths
    if params["checkpoint_path"] is None:
        # Build log_id based on model type
        log_id_parts = [str(cfg.model_cfg.name).lower()]
        
        # Add model-specific parameters
        if cfg.model_cfg.name.lower() in ['bubble_ddpm', 'flow_matching', 'flow_matching_jit', 'flow_matching_ar', 'flow_matching_ar_bootstrap', 'edm']:
            # DDPM and Flow Matching models (similar architectures)
            log_id_parts.extend([
                str(cfg.model_cfg.get('base_channels', 'NA')),
                str(cfg.model_cfg.get('time_embed_dim', 'NA')),
                str(cfg.model_cfg.get('num_res_blocks', 'NA')),
                str(cfg.model_cfg.get('use_attention', 'NA'))
            ])
            # Add schedule type for DDPM models
            if cfg.model_cfg.name.lower() == 'bubble_ddpm':
                schedule_type = cfg.model_cfg.get('schedule_type', 'linear')
                if schedule_type != 'linear':
                    log_id_parts.append(f"sched_{schedule_type}")
            # Add noise_scale for flow matching models (important for normalized data)
            if cfg.model_cfg.name.lower() in ['flow_matching', 'flow_matching_jit', 'flow_matching_ar', 'flow_matching_ar_bootstrap']:
                noise_scale = cfg.model_cfg.get('noise_scale', 1.0)
                if noise_scale != 1.0:
                    log_id_parts.append(f"ns{noise_scale}")
            # Add AR model indicator and features
            if cfg.model_cfg.name.lower() == 'flow_matching_ar':
                log_id_parts.append("ar")
                # Add residual prediction indicator if enabled
                if cfg.model_cfg.get('residual_prediction', False):
                    log_id_parts.append("residual")
                # Add scheduled sampling indicator if enabled
                ss_cfg = cfg.model_cfg.get('scheduled_sampling', {})
                if ss_cfg.get('enabled', False):
                    log_id_parts.append("ss")
                # Add auxiliary losses indicators
                aux_losses = cfg.model_cfg.get('auxiliary_losses', {})
                if aux_losses.get('spectral_enabled', False):
                    log_id_parts.append("spectral")
                if aux_losses.get('gradient_enabled', False):
                    log_id_parts.append("grad")
                if aux_losses.get('divergence_enabled', False):
                    log_id_parts.append("div")
                if aux_losses.get('vorticity_enabled', False):
                    log_id_parts.append("vort")
                if aux_losses.get('advection_enabled', False):
                    log_id_parts.append("adv")
            # Add AR Bootstrap model indicator and features
            if cfg.model_cfg.name.lower() == 'flow_matching_ar_bootstrap':
                log_id_parts.append(f"hist{cfg.model_cfg.get('history_length', 10)}")
                log_id_parts.append(f"roll{cfg.model_cfg.get('rollout_length', 5)}")
                # Add history encoder type indicator (conv3d or tmix for temporal_mixer)
                he_type = cfg.model_cfg.get('history_encoder_type', 'conv3d')
                if he_type == 'temporal_mixer':
                    log_id_parts.append("tmix")
                    # Add TemporalMixer-specific settings
                    if cfg.model_cfg.get('temporal_mixer_spatial_conv', True):
                        log_id_parts.append("spatial")
                    if cfg.model_cfg.get('temporal_mixer_temporal_weights', True):
                        log_id_parts.append("tweights")
                elif he_type == 'attention':
                    log_id_parts.append("attn")
                    log_id_parts.append(f"d{cfg.model_cfg.get('attention_encoder_embed_dim', 256)}")
                    log_id_parts.append(f"L{cfg.model_cfg.get('attention_encoder_depth', 4)}")
                    log_id_parts.append(f"p{cfg.model_cfg.get('attention_encoder_patch_size', 8)}")
                else:
                    log_id_parts.append("conv3d")
                    log_id_parts.append(f"hid{cfg.model_cfg.get('history_encoder_hidden', 64)}")
                # Add scheduled sampling indicator if enabled
                ss_cfg = cfg.model_cfg.get('scheduled_sampling', {})
                if ss_cfg.get('enabled', False):
                    log_id_parts.append("ss")
                # Add push forward indicator if enabled
                pf_cfg = cfg.model_cfg.get('push_forward', {})
                if pf_cfg.get('enabled', False):
                    log_id_parts.append("pf")
                    log_id_parts.append(f"max{pf_cfg.get('max_push_steps', 3)}")
                # Add AR temporal decay indicator if enabled
                ar_decay_cfg = cfg.model_cfg.get('ar_temporal_decay', {})
                if ar_decay_cfg.get('enabled', False):
                    log_id_parts.append("decay")
                # Add auxiliary losses indicators
                aux_losses = cfg.model_cfg.get('auxiliary_losses', {})
                if aux_losses.get('spectral_enabled', False):
                    log_id_parts.append("spectral")
                if aux_losses.get('gradient_enabled', False):
                    log_id_parts.append("grad")           
                # Add push forward indicator if enabled
                pf_cfg = cfg.model_cfg.get('push_forward', {})
                if pf_cfg.get('enabled', False):
                    log_id_parts.append("pf")
                    log_id_parts.append(f"max{pf_cfg.get('max_push_steps', 3)}")
                # Add AR temporal decay indicator if enabled
                ar_decay_cfg = cfg.model_cfg.get('ar_temporal_decay', {})
                if ar_decay_cfg.get('enabled', False):
                    log_id_parts.append("decay")
                # Add auxiliary losses indicators
                aux_losses = cfg.model_cfg.get('auxiliary_losses', {})
                if aux_losses.get('spectral_enabled', False):
                    log_id_parts.append("spectral")
                if aux_losses.get('gradient_enabled', False):
                    log_id_parts.append("grad")
        elif cfg.model_cfg.name.lower() == 've_sde':
            # VE-SDE Score-Based model
            log_id_parts.extend([
                str(cfg.model_cfg.get('base_channels', 'NA')),
                str(cfg.model_cfg.get('sigma_embed_dim', 'NA')),
                str(cfg.model_cfg.get('num_res_blocks', 'NA')),
                str(cfg.model_cfg.get('use_attention', 'NA'))
            ])
        elif cfg.model_cfg.name.lower() == 'unet':
            # UNet model (direct regression)
            log_id_parts.append(str(cfg.model_cfg.get('init_features', 32)))
        elif cfg.model_cfg.name.lower() == 'ffno':
            # FFNO model (Factorized Fourier Neural Operator)
            log_id_parts.extend([
                f"m{cfg.model_cfg.get('modes', 12)}",
                f"w{cfg.model_cfg.get('width', 64)}",
                f"l{cfg.model_cfg.get('n_layers', 4)}"
            ])
        elif cfg.model_cfg.name.lower() == 'unet_ar':
            # Autoregressive UNet model (direct regression, no diffusion)
            log_id_parts.extend([
                str(cfg.model_cfg.get('init_features', 'NA')),
                "ar"  # Always add AR indicator
            ])
            # Add residual prediction indicator if enabled
            if cfg.model_cfg.get('residual_prediction', False):
                log_id_parts.append("residual")
            # Add scheduled sampling indicator if enabled
            ss_cfg = cfg.model_cfg.get('scheduled_sampling', {})
            if ss_cfg.get('enabled', False):
                log_id_parts.append("ss")
            # Add auxiliary losses indicators
            aux_losses = cfg.model_cfg.get('auxiliary_losses', {})
            if aux_losses.get('spectral_enabled', False):
                log_id_parts.append("spectral")
            if aux_losses.get('gradient_enabled', False):
                log_id_parts.append("grad")
        elif cfg.model_cfg.name.lower() == 'edm':
            # EDM model (EDM-style diffusion baseline)
            # Configured to match flow_matching for fair comparison
            log_id_parts.extend([
                f"ch{cfg.model_cfg.get('model_channels', 32)}",
                f"b{cfg.model_cfg.get('num_blocks', 2)}",
                f"s{cfg.model_cfg.get('num_sampling_steps', 50)}"
            ])
        elif cfg.model_cfg.name.lower() == 'edm_ar_bootstrap':
            # EDM AR Bootstrap model
            log_id_parts.extend([
                f"ch{cfg.model_cfg.get('model_channels', 32)}",
                f"b{cfg.model_cfg.get('num_blocks', 2)}",
                f"hist{cfg.model_cfg.get('history_length', 10)}",
                f"roll{cfg.model_cfg.get('rollout_length', 5)}",
            ])
            he_type = cfg.model_cfg.get('history_encoder_type', 'conv3d')
            if he_type == 'temporal_mixer':
                log_id_parts.append("tmix")
            elif he_type == 'attention':
                log_id_parts.append("attn")
                log_id_parts.append(f"d{cfg.model_cfg.get('attention_encoder_embed_dim', 256)}")
                log_id_parts.append(f"L{cfg.model_cfg.get('attention_encoder_depth', 4)}")
                log_id_parts.append(f"p{cfg.model_cfg.get('attention_encoder_patch_size', 8)}")
            else:
                log_id_parts.append("conv3d")
            ss_cfg = cfg.model_cfg.get('scheduled_sampling', {})
            if ss_cfg.get('enabled', False):
                log_id_parts.append("ss")
        elif cfg.model_cfg.name.lower() == 'diffusionpde':
            # DiffusionPDE baseline (unconditional joint diffusion + guided sampling)
            log_id_parts.extend([
                f"ch{cfg.model_cfg.get('model_channels', 32)}",
                f"b{cfg.model_cfg.get('num_blocks', 2)}",
                f"s{cfg.model_cfg.get('num_sampling_steps', 50)}",
                f"zobs{cfg.model_cfg.get('zeta_obs', 1.0)}",
                f"zpde{cfg.model_cfg.get('zeta_pde', 0.5)}",
            ])
        else:
            # Generic model
            log_id_parts.append('default')
        
        # Add task name if available
        if params["task_cfg"] is not None:
            log_id_parts.append(str(params["task_cfg"].get('name', 'unknown_task')).lower())
        
        # Add normalization mode and loss type to run name
        run_norm_mode = OmegaConf.select(cfg, 'norm_mode', default='all')
        if run_norm_mode != 'all':
            log_id_parts.append(f"norm_{run_norm_mode}")
        run_loss_type = cfg.model_cfg.get('loss_type', 'mse')
        if run_loss_type != 'mse':
            if run_loss_type == 'hybrid':
                weights = cfg.model_cfg.get('loss_weights', {})
                parts = [f"{n}{w}" for n, w in weights.items()]
                log_id_parts.append("loss_" + "+".join(parts))
            else:
                log_id_parts.append(f"loss_{run_loss_type}")
        
        # Add common parameters
        downsample_factor_for_name = cfg.data_cfg.get('downsample_factor', 1)
        downsample_str = f"ds{downsample_factor_for_name}" if downsample_factor_for_name > 1 else "fullres"
        log_id_parts.extend([
            str(cfg.data_cfg.dataset).lower(),
            str(cfg.model_cfg.get('conditioning_strategy', 'none')).lower(),
            downsample_str,
            str(os.getenv("SLURM_JOB_ID"))
        ])
        
        log_id = "_".join(log_id_parts)
        params["log_dir"] = os.path.join(cfg.log_dir, log_id)
        os.makedirs(params["log_dir"], exist_ok=True)
        
        # Check for auto-resume
        if params["auto_resume"] and params["resume_from_last"]:
            found_checkpoint = find_latest_checkpoint(params["log_dir"])
            if found_checkpoint:
                params["checkpoint_path"] = found_checkpoint
                if is_leader_process():
                    print(f"Auto-resuming from checkpoint: {found_checkpoint}")
        
        preempt_ckpt_path = params["log_dir"] + "/hpc_ckpt_1.ckpt"
    else:
        # Explicit checkpoint path provided
        if os.path.isfile(params["checkpoint_path"]):
            # Extract log directory from checkpoint path
            checkpoint_dir = os.path.dirname(params["checkpoint_path"])
            # Go up to find the main log directory (handle different checkpoint locations)
            if "checkpoints" in checkpoint_dir:
                # Standard Lightning checkpoint structure
                params["log_dir"] = os.path.dirname(checkpoint_dir)
            elif "lightning_logs" in checkpoint_dir:
                # Lightning logs structure
                params["log_dir"] = checkpoint_dir.split("lightning_logs")[0].rstrip("/")
            else:
                # Direct checkpoint in log dir
                params["log_dir"] = checkpoint_dir
            
            # Generate log_id from log directory
            log_id = os.path.basename(params["log_dir"])
            
            # Set preemption checkpoint path
            preempt_ckpt_path = params["log_dir"] + "/hpc_ckpt_resume.ckpt"
            
            if is_leader_process():
                print(f"Resuming from explicit checkpoint: {params['checkpoint_path']}")
                print(f"Log directory set to: {params['log_dir']}")
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {params['checkpoint_path']}")

    # Initialize loggers
    loggers = [CSVLogger(save_dir=params["log_dir"])]
    
    # Initialize wandb if enabled and we're the leader process
    if params["use_wandb"] and is_leader_process():
        # Generate wandb run name if not provided
        wandb_name = params["wandb_cfg"].get("name", None)
        if wandb_name is None:
            wandb_name = log_id
        
        # Check for existing WandB run info (for resuming)
        wandb_info_file = os.path.join(params["log_dir"], "wandb_run_info.json")
        wandb_run_id = None
        resume_mode = None
        
        # If resuming from checkpoint (either manual or auto), try to continue the WandB run
        if os.path.exists(wandb_info_file):
            try:
                with open(wandb_info_file, 'r') as f:
                    wandb_info = json.load(f)
                wandb_run_id = wandb_info.get('id')
                saved_run_name = wandb_info.get('name', 'unknown')
                resume_mode = "allow"  # "allow" continues if exists, creates new if not
                print(f"📊 Resuming WandB run:")
                print(f"   Name: {saved_run_name}")
                print(f"   ID: {wandb_run_id}")
                print(f"   URL: https://wandb.ai/{params['wandb_cfg'].get('entity', 'YOUR_ENTITY')}/{params['wandb_cfg'].get('project', 'bubblefusion')}/runs/{wandb_run_id}")
            except Exception as e:
                print(f"⚠️  Could not read WandB run info: {e}")
                wandb_run_id = None
        
        # Initialize wandb
        wandb_init_kwargs = {
            "project": params["wandb_cfg"].get("project", "bubblefusion"),
            "entity": params["wandb_cfg"].get("entity", None),
            "name": wandb_name,
            "tags": params["wandb_cfg"].get("tags", []),
            "notes": params["wandb_cfg"].get("notes", ""),
            "config": {
                "model": dict(params["model_cfg"]),
                "data": dict(params["data_cfg"]),
                "optimizer": dict(params["optim_cfg"]),
                "scheduler": dict(params["scheduler_cfg"]),
                "task": dict(params["task_cfg"]) if params["task_cfg"] is not None else {},
                "task_name": params["task_cfg"].get('name', 'unknown') if params["task_cfg"] is not None else 'none',
                "conditioning_channels": list(params["task_cfg"].get('conditioning_channels', [])) if params["task_cfg"] is not None else [],
                "target_channels": list(params["task_cfg"].get('target_channels', [])) if params["task_cfg"] is not None else [],
                "conditioning_names": list(params["task_cfg"].get('conditioning_names', [])) if params["task_cfg"] is not None else [],
                "target_names": list(params["task_cfg"].get('target_names', [])) if params["task_cfg"] is not None else [],
                "batch_size": cfg.batch_size,
                "max_epochs": cfg.max_epochs,
                "nodes": params["nodes"],
                "devices": params["devices"],
                "seed": cfg.seed,
                # AR model specific configs (for easy filtering in wandb)
                "residual_prediction": cfg.model_cfg.get('residual_prediction', False),
                "scheduled_sampling": cfg.model_cfg.get('scheduled_sampling', {}).get('enabled', False),
                "spectral_loss_enabled": cfg.model_cfg.get('auxiliary_losses', {}).get('spectral_enabled', False),
                "gradient_loss_enabled": cfg.model_cfg.get('auxiliary_losses', {}).get('gradient_enabled', False),
                # Physics-informed losses
                "divergence_loss_enabled": cfg.model_cfg.get('auxiliary_losses', {}).get('divergence_enabled', False),
                "divergence_loss_weight": cfg.model_cfg.get('auxiliary_losses', {}).get('divergence_weight', 0.1),
                "vorticity_loss_enabled": cfg.model_cfg.get('auxiliary_losses', {}).get('vorticity_enabled', False),
                "vorticity_loss_weight": cfg.model_cfg.get('auxiliary_losses', {}).get('vorticity_weight', 0.1),
                "advection_loss_enabled": cfg.model_cfg.get('auxiliary_losses', {}).get('advection_enabled', False),
                "advection_loss_weight": cfg.model_cfg.get('auxiliary_losses', {}).get('advection_weight', 0.1),
                "advection_dt": cfg.model_cfg.get('auxiliary_losses', {}).get('advection_dt', 0.001),
                # Flow matching noise scale (important for normalized data)
                "noise_scale": cfg.model_cfg.get('noise_scale', 1.0),
                # AR Bootstrap specific configs
                "history_encoder_type": cfg.model_cfg.get('history_encoder_type', 'conv3d'),
                "history_length": cfg.model_cfg.get('history_length', 10),
                "rollout_length": cfg.model_cfg.get('rollout_length', 5),
                "temporal_mixer_spatial_conv": cfg.model_cfg.get('temporal_mixer_spatial_conv', True),
                "temporal_mixer_temporal_weights": cfg.model_cfg.get('temporal_mixer_temporal_weights', True),
                "ar_temporal_decay": cfg.model_cfg.get('ar_temporal_decay', {}).get('enabled', False),
                # Push forward trick configs
                "push_forward_enabled": cfg.model_cfg.get('push_forward', {}).get('enabled', False),
                "push_forward_max_steps": cfg.model_cfg.get('push_forward', {}).get('max_push_steps', 3),
            },
            "dir": params["log_dir"]
        }
        
        # Add resume parameters if resuming
        if wandb_run_id is not None:
            wandb_init_kwargs["id"] = wandb_run_id
            wandb_init_kwargs["resume"] = resume_mode
        
        run = wandb.init(**wandb_init_kwargs)
        
        # Save the WandB run info (ID, name, URL) for future resumption
        if not os.path.exists(wandb_info_file):
            try:
                wandb_info = {
                    'id': run.id,
                    'name': run.name,
                    'url': run.url,
                    'project': run.project,
                    'entity': run.entity
                }
                with open(wandb_info_file, 'w') as f:
                    json.dump(wandb_info, f, indent=2)
                print(f"💾 Saved WandB run info:")
                print(f"   Name: {run.name}")
                print(f"   ID: {run.id}")
                print(f"   URL: {run.url}")
                print(f"   File: {wandb_info_file}")
            except Exception as e:
                print(f"⚠️  Could not save WandB run info: {e}")
        
        # Add wandb logger to loggers list
        # IMPORTANT: Pass experiment=run to reuse the existing wandb run
        # instead of potentially creating a new one
        wandb_logger = WandbLogger(
            experiment=run,  # Use the existing wandb run we just initialized
            save_dir=params["log_dir"]
        )
        loggers.append(wandb_logger)

    if is_leader_process():
        pprint.pprint(params)

    # Check if model uses wall temperature conditioning
    is_ve_sde_model = cfg.model_cfg.name.lower() == 've_sde'
    is_ar_model = cfg.model_cfg.name.lower() in ['flow_matching_ar', 'unet_ar']
    is_ar_bootstrap_model = cfg.model_cfg.name.lower() in ['flow_matching_ar_bootstrap', 'edm_ar_bootstrap']
    is_history_model = cfg.model_cfg.name.lower() == 'flow_matching_history'
    conditioning_strategy = cfg.model_cfg.get('conditioning_strategy', 'none')
    use_wall_temp = (conditioning_strategy != 'none')
    
    # Extract noise configuration from task_cfg for Task 3 (noisy_velocity_from_interface)
    # This enables optical flow noise simulation on conditioning inputs
    noise_cfg = None
    if params["task_cfg"] is not None:
        noise_cfg = params["task_cfg"].get('noise_cfg', None)
        if noise_cfg is not None and noise_cfg.get('enabled', True):
            if is_leader_process():
                noise_type = noise_cfg.get('noise_type', 'optical_flow')
                print(f"\n🔊 Task 3 Noise Injection: ENABLED (type: {noise_type})")
                if noise_type in ['gaussian', 'simple']:
                    print(f"   SDF noise std: {noise_cfg.get('sdf_noise_std', 0.1)}")
                    print(f"   Velocity noise std: {noise_cfg.get('vel_noise_std', 0.05)}")
                else:  # optical_flow / complex
                    print(f"   SDF noise: std={noise_cfg.get('sdf_noise_std', 0.1)}, gradient_scale={noise_cfg.get('sdf_gradient_scale', 0.3)}")
                    print(f"   Velocity noise: base_std={noise_cfg.get('vel_base_noise_std', 0.05)}, scale_factor={noise_cfg.get('vel_scale_factor', 0.15)}")
                    print(f"   Correlation length: {noise_cfg.get('correlation_length', 3.0)} pixels")
    
    # Get downsample factor for fast prototyping (1 = no downsampling)
    downsample_factor = cfg.data_cfg.get('downsample_factor', 1)
    if downsample_factor > 1 and is_leader_process():
        original_res = 512  # Typical original resolution
        new_res = original_res // downsample_factor
        print(f"\n📐 Fast Prototyping Mode: Downsampling {downsample_factor}x")
        print(f"   Resolution: {original_res}x{original_res} → {new_res}x{new_res}")
    
    # Compute normalization statistics from training files ONCE
    # This ensures consistent normalization across training and validation datasets
    # Priority: 1) Explicit file path, 2) Previous training stats, 3) Compute new
    normalization_stats = None
    
    # Option 1: Load from explicitly provided file (via config or command line)
    # Usage: python scripts/train.py +normalization_stats=/path/to/stats.json
    stats_file_path = OmegaConf.select(cfg, 'normalization_stats', default=None)
    if stats_file_path is not None:
        stats_file_path = str(stats_file_path)  # Convert OmegaConf string if needed
        if is_leader_process():
            print(f"\n📁 Found normalization_stats path in config: {stats_file_path}")
            if os.path.exists(stats_file_path):
                print(f"   ✓ File exists!")
            else:
                print(f"   ✗ File NOT found at this path!")
    if stats_file_path and os.path.exists(stats_file_path):
        if is_leader_process():
            print("\n" + "=" * 60)
            print("📊 Loading Normalization Statistics from Provided File")
            print("=" * 60)
        with open(stats_file_path, 'r') as f:
            normalization_stats = json.load(f)
        if is_leader_process():
            print(f"   ✓ Loaded from: {stats_file_path}")
            print(f"   Temperature: [{normalization_stats['temperature']['min']:.1f}, {normalization_stats['temperature']['max']:.1f}]°C")
            print(f"   Velocity scale: {normalization_stats['unified_velocity_scale']:.4f}")
            print(f"   SDF scale: {normalization_stats['sdf']['scale']:.4f}")
            print("=" * 60)
    
    # Option 2: Try to load from previous training (if resuming)
    if normalization_stats is None and params.get("checkpoint_path") and os.path.exists(params["checkpoint_path"]):
        # Check if normalization_stats.json exists in the checkpoint directory
        checkpoint_dir = os.path.dirname(params["checkpoint_path"])
        if "checkpoints" in checkpoint_dir:
            checkpoint_dir = os.path.dirname(checkpoint_dir)
        stats_file = os.path.join(checkpoint_dir, "normalization_stats.json")
        
        if os.path.exists(stats_file):
            if is_leader_process():
                print("\n" + "=" * 60)
                print("📊 Loading Normalization Statistics from Previous Training")
                print("=" * 60)
            with open(stats_file, 'r') as f:
                normalization_stats = json.load(f)
            if is_leader_process():
                print(f"   ✓ Loaded from: {stats_file}")
                print(f"   Temperature: [{normalization_stats['temperature']['min']:.1f}, {normalization_stats['temperature']['max']:.1f}]°C")
                print(f"   Velocity scale: {normalization_stats['unified_velocity_scale']:.4f}")
                print(f"   SDF scale: {normalization_stats['sdf']['scale']:.4f}")
                print("=" * 60)
    
    # Option 3: Compute stats if not loaded
    if normalization_stats is None:
        if is_leader_process():
            print("\n" + "=" * 60)
            print("📊 Computing Normalization Statistics from Training Data")
            print("=" * 60)
        
        normalization_stats = compute_normalization_stats(
            filenames=cfg.data_cfg.train_paths,
            start_time=cfg.data_cfg.start_time,
            verbose=is_leader_process()
        )
    
     # Add downsample_factor to normalization_stats for physics losses
    # This allows the model to compute correct grid spacing for derivatives
    normalization_stats['downsample_factor'] = downsample_factor
    
    # Read norm_mode: 'all' (default), 'none', or 'temperature_only'
    norm_mode = OmegaConf.select(cfg, 'norm_mode', default='all')
    
    if is_leader_process():
        print(f"\n✓ Normalization mode: {norm_mode}")
        if norm_mode == 'none':
            print(f"   ALL fields passed through RAW (no normalization)")
        elif norm_mode == 'temperature_only':
            print(f"   Temperature: tanh to [-1, 1], range [{normalization_stats['temperature']['min']:.1f}, {normalization_stats['temperature']['max']:.1f}]°C")
            print(f"   Velocity/SDF: RAW (no normalization)")
        else:
            print(f"   Temperature: tanh to [-1, 1], range [{normalization_stats['temperature']['min']:.1f}, {normalization_stats['temperature']['max']:.1f}]°C")
            print(f"   Velocity: unified scale = {normalization_stats['unified_velocity_scale']:.4f}")
            print(f"   SDF: zero-preserving scale = {normalization_stats['sdf']['scale']:.4f}")
        print(f"   Downsample factor: {normalization_stats.get('downsample_factor', 1)}")
        print("=" * 60)
        
        # Save normalization stats to JSON file for inference and future retraining
        stats_file = os.path.join(params["log_dir"], "normalization_stats.json")
        # Convert numpy types to native Python types for JSON serialization
        stats_serializable = {}
        for key, value in normalization_stats.items():
            if isinstance(value, dict):
                stats_serializable[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                           for k, v in value.items()}
            elif isinstance(value, (np.floating, np.integer)):
                stats_serializable[key] = float(value)
            else:
                stats_serializable[key] = value
        with open(stats_file, 'w') as f:
            json.dump(stats_serializable, f, indent=2)
        print(f"💾 Saved normalization stats to: {stats_file}")
        print(f"   (Use this file for inference and retraining on the same dataset)")
    
    # Create dataset based on model type
    if is_ar_bootstrap_model:
        # Bootstrap AR model uses BulkFlowARBootstrap dataset
        history_length = cfg.model_cfg.get('history_length', 10)
        history_stride = cfg.model_cfg.get('history_stride', 1)
        rollout_length = cfg.model_cfg.get('rollout_length', 5)
        
        if is_leader_process():
            print(f"\n🚀 Using Bootstrap AR Dataset:")
            print(f"   History length: {history_length} (for bootstrap initialization)")
            print(f"   History stride: {history_stride} (spans {history_length * history_stride} timesteps)")
            print(f"   Rollout length: {rollout_length} (trajectory segment)")
            print(f"   ℹ️  First frame uses bootstrap mode, rest uses AR mode")
        
        train_dataset = BulkFlowARBootstrap(
                    filenames=cfg.data_cfg.train_paths,
                    output_fields=cfg.data_cfg.output_fields,
                    start_time=cfg.data_cfg.start_time,
                    history_length=history_length,
                    rollout_length=rollout_length,
                    normalization_stats=normalization_stats,
                    return_wall_temp=use_wall_temp,
                    noise_cfg=noise_cfg,
                    downsample_factor=downsample_factor,
                    norm_mode=norm_mode
                )
    elif is_ar_model:
        # Autoregressive model uses BulkFlowAutoregressive dataset
        # Check if scheduled sampling is enabled
        ss_cfg = cfg.model_cfg.get('scheduled_sampling', {})
        scheduled_sampling_enabled = ss_cfg.get('enabled', False)
        
        if is_leader_process():
            if scheduled_sampling_enabled:
                print(f"\n🔄 Using Autoregressive Dataset (Scheduled Sampling):")
                print(f"   Input: [conditioning_t, output_(t-1)]")
                print(f"   Target: output_t")
                print(f"   Extra context: [conditioning_(t-1), output_(t-2)]")
                print(f"   📊 Schedule: {ss_cfg.get('schedule_type', 'linear')}")
                print(f"   📊 Warmup: {ss_cfg.get('warmup_epochs', 5)} epochs")
                print(f"   📊 Transition: {ss_cfg.get('transition_epochs', 40)} epochs")
            else:
                print(f"\n🔄 Using Autoregressive Dataset (Teacher Forcing):")
                print(f"   Input: [conditioning_t, output_(t-1)]")
                print(f"   Target: output_t")
        
        train_dataset = BulkFlowAutoregressive(
                    filenames=cfg.data_cfg.train_paths,
                    output_fields=cfg.data_cfg.output_fields,
                    start_time=cfg.data_cfg.start_time,
                    normalization_stats=normalization_stats,
                    return_wall_temp=use_wall_temp,
                    noise_cfg=noise_cfg,
                    downsample_factor=downsample_factor,
                    scheduled_sampling=scheduled_sampling_enabled,
                    norm_mode=norm_mode
                )
    elif is_history_model:
        history_window = cfg.model_cfg.get('history_window', 10)
        history_stride = int(cfg.model_cfg.get('history_stride', 1))
        if is_leader_process():
            print(f"\n📊 Using History-Window Dataset:")
            print(f"   History window (W): {history_window}")
            print(f"   History stride (S): {history_stride}")
            print(f"   Input: [SDF_{{t-(W-1)*S}}, iVel_{{t-(W-1)*S}}, ..., SDF_t, iVel_t]")
            print(f"   Target: [velx_t, vely_t, temp_t]")

        train_dataset = BulkFlowHistory(
                    filenames=cfg.data_cfg.train_paths,
                    output_fields=cfg.data_cfg.output_fields,
                    start_time=cfg.data_cfg.start_time,
                    history_window=history_window,
                    history_stride=history_stride,
                    normalization_stats=normalization_stats,
                    return_wall_temp=use_wall_temp,
                    noise_cfg=noise_cfg,
                    downsample_factor=downsample_factor,
                    norm_mode=norm_mode,
                )
    else:
        train_dataset = BulkFlow(
                    filenames=cfg.data_cfg.train_paths,
                    output_fields=cfg.data_cfg.output_fields,
                    start_time=cfg.data_cfg.start_time,
                    normalization_stats=normalization_stats,
                    return_wall_temp=use_wall_temp,
                    noise_cfg=noise_cfg,
                    downsample_factor=downsample_factor,
                    norm_mode=norm_mode
                )
    
    # For Task 3 (noisy inputs), create two validation datasets:
    # 1. val_noisy: Same noise as training (deployment performance)
    # 2. val_clean: No noise (physics fidelity check)
    use_dual_validation = noise_cfg is not None and noise_cfg.get('enabled', True)
    
    # Select dataset class based on model type
    if is_ar_bootstrap_model:
        DatasetClass = BulkFlowARBootstrap
    elif is_ar_model:
        DatasetClass = BulkFlowAutoregressive
    elif is_history_model:
        DatasetClass = BulkFlowHistory
    else:
        DatasetClass = BulkFlow
    
    # Common dataset args
    # Use the same normalization_stats computed from training data
    base_dataset_args = {
        'filenames': cfg.data_cfg.val_paths,
        'output_fields': cfg.data_cfg.output_fields,
        'start_time': cfg.data_cfg.start_time,
        'normalization_stats': normalization_stats,
        'return_wall_temp': use_wall_temp,
        'downsample_factor': downsample_factor,
        'norm_mode': norm_mode,
    }
    
    # Add bootstrap args if using bootstrap model
    if is_ar_bootstrap_model:
        base_dataset_args['history_length'] = cfg.model_cfg.get('history_length', 10)
        base_dataset_args['history_stride'] = cfg.model_cfg.get('history_stride', 1)
        base_dataset_args['rollout_length'] = cfg.model_cfg.get('rollout_length', 5)
    
    # Add history_window arg for history model
    if is_history_model:
        base_dataset_args['history_window'] = cfg.model_cfg.get('history_window', 10)
        base_dataset_args['history_stride'] = int(cfg.model_cfg.get('history_stride', 1))

    # Add scheduled_sampling flag for AR model
    # Note: For validation, we always use teacher forcing (scheduled_sampling=False)
    # because we want to measure pure model quality without the mixing
    if is_ar_model:
        base_dataset_args['scheduled_sampling'] = False  # Always teacher forcing for validation
    
    if use_dual_validation:
        if is_leader_process():
            print(f"\n📊 Dual Validation Setup for Task 3:")
            print(f"   val_noisy: Same noise as training (deployment metric)")
            print(f"   val_clean: No noise (physics fidelity metric)")
        
        # Validation with noise (deployment performance)
        val_dataset_noisy = DatasetClass(
                    **base_dataset_args,
                    noise_cfg=noise_cfg  # WITH noise
                )
        
        # Validation without noise (physics fidelity)
        val_dataset_clean = DatasetClass(
                    **base_dataset_args,
                    noise_cfg=None  # NO noise
                )
    else:
        # Standard single validation (no noise or Task 1/2)
        val_dataset_noisy = None
        val_dataset_clean = DatasetClass(
                    **base_dataset_args,
                    noise_cfg=noise_cfg
                )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    
    # Create validation dataloaders
    if use_dual_validation:
        val_dataloader_noisy = DataLoader(
            val_dataset_noisy,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        val_dataloader_clean = DataLoader(
            val_dataset_clean,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        # Combined validation dataloaders for Lightning
        # Order matters: dataloader_idx=0 → clean, dataloader_idx=1 → noisy
        # This matches model's validation_step which expects is_clean_val = (dataloader_idx == 0)
        val_dataloader = {"clean": val_dataloader_clean, "noisy": val_dataloader_noisy}
    else:
        val_dataloader = DataLoader(
            val_dataset_clean,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    # Print dataset sample shapes
    # Get the appropriate validation dataloader for sampling
    if use_dual_validation:
        sample_val_dataloader = val_dataloader["clean"]  # Use clean for shape printing
    else:
        sample_val_dataloader = val_dataloader
    
    # Print dataset sample shapes
    if is_ar_bootstrap_model:
        # BulkFlowARBootstrap returns: (conditioning_history, conditioning_sequence, target_sequence, [wall_temp])
        train_batch = next(iter(train_dataloader))
        
        if use_wall_temp:
            cond_hist, cond_seq, target_seq, train_wall_temp = train_batch
            print("Train dataset sample (bootstrap AR):")
            print(f"   Conditioning history: {cond_hist.shape} [B, T_hist, C_cond, H, W]")
            print(f"   Conditioning sequence: {cond_seq.shape} [B, L, C_cond, H, W]")
            print(f"   Target sequence: {target_seq.shape} [B, L, C_out, H, W]")
            print(f"   Wall temp: {train_wall_temp.shape}")
        else:
            cond_hist, cond_seq, target_seq = train_batch
            print("Train dataset sample (bootstrap AR):")
            print(f"   Conditioning history: {cond_hist.shape} [B, T_hist, C_cond, H, W]")
            print(f"   Conditioning sequence: {cond_seq.shape} [B, L, C_cond, H, W]")
            print(f"   Target sequence: {target_seq.shape} [B, L, C_out, H, W]")
        print(f"   ℹ️  First frame uses bootstrap mode, rest uses AR mode")
        print(f"🌡️  Wall temperature conditioning: NONE")
    elif is_ar_model:
        # BulkFlowAutoregressive returns different values based on scheduled_sampling mode
        # Teacher forcing: (inp_t, prev_output, out_t, [wall_temp])
        # Scheduled sampling: (inp_t, prev_output, out_t, cond_t-1, out_t-2, [wall_temp])
        train_batch = next(iter(train_dataloader))
        num_elements = len(train_batch)
        
        if scheduled_sampling_enabled:
            # Scheduled sampling mode: 5 or 6 elements
            if use_wall_temp:
                train_inp_t, train_prev_out, train_out_t, train_cond_tm1, train_out_tm2, train_wall_temp = train_batch
                print("Train dataset sample (autoregressive with scheduled sampling):")
                print(f"   Conditioning (t): {train_inp_t.shape}")
                print(f"   Previous output (t-1, GT): {train_prev_out.shape}")
                print(f"   Target output (t): {train_out_t.shape}")
                print(f"   Conditioning (t-1): {train_cond_tm1.shape}")
                print(f"   Output (t-2): {train_out_tm2.shape}")
                print(f"   Wall temp: {train_wall_temp.shape}")
            else:
                train_inp_t, train_prev_out, train_out_t, train_cond_tm1, train_out_tm2 = train_batch
                print("Train dataset sample (autoregressive with scheduled sampling):")
                print(f"   Conditioning (t): {train_inp_t.shape}")
                print(f"   Previous output (t-1, GT): {train_prev_out.shape}")
                print(f"   Target output (t): {train_out_t.shape}")
                print(f"   Conditioning (t-1): {train_cond_tm1.shape}")
                print(f"   Output (t-2): {train_out_tm2.shape}")
        else:
            # Teacher forcing mode: 3 or 4 elements
            if use_wall_temp:
                train_inp_t, train_prev_out, train_out_t, train_wall_temp = train_batch
                print("Train dataset sample (autoregressive with teacher forcing):")
                print(f"   Conditioning (t): {train_inp_t.shape}")
                print(f"   Previous output (t-1, teacher forcing): {train_prev_out.shape}")
                print(f"   Target output (t): {train_out_t.shape}")
                print(f"   Wall temp: {train_wall_temp.shape}")
            else:
                train_inp_t, train_prev_out, train_out_t = train_batch
                print("Train dataset sample (autoregressive with teacher forcing):")
                print(f"   Conditioning (t): {train_inp_t.shape}")
                print(f"   Previous output (t-1, teacher forcing): {train_prev_out.shape}")
                print(f"   Target output (t): {train_out_t.shape}")
        print(f"🌡️  Wall temperature conditioning: NONE (baseline for ablation study)")
    elif use_wall_temp:
        train_ip_sample, train_op_sample, train_wall_temp = next(iter(train_dataloader))
        print("Train dataset sample:", train_ip_sample.shape, train_op_sample.shape, "Wall temp:", train_wall_temp.shape)
        val_ip_sample, val_op_sample, val_wall_temp = next(iter(sample_val_dataloader))
        print("Validation dataset sample:", val_ip_sample.shape, val_op_sample.shape, "Wall temp:", val_wall_temp.shape)
        print(f"🌡️  Wall temperature conditioning: {conditioning_strategy.upper()}")
        print(f"   Example wall temperatures (train): {train_wall_temp[:min(4, len(train_wall_temp))].tolist()}")
    else:
        train_ip_sample, train_op_sample = next(iter(train_dataloader))
        print("Train dataset sample:", train_ip_sample.shape, train_op_sample.shape)
        val_ip_sample, val_op_sample = next(iter(sample_val_dataloader))
        print("Validation dataset sample:", val_ip_sample.shape, val_op_sample.shape)
        print(f"🌡️  Wall temperature conditioning: NONE (baseline for ablation study)")

    # Initialize model
    # Print task configuration if available
    if params["task_cfg"] is not None:
        if is_leader_process():
            print(f"\n📋 Task Configuration: {params['task_cfg'].get('name', 'unknown')}")
            print(f"   Description: {params['task_cfg'].get('description', 'N/A')}")
    
    if cfg.model_cfg.name.lower() == 'bubble_ddpm':
        # Pass task_cfg and normalization_stats to DDPM model
        model = BubbleDDPMLightning(cfg.model_cfg, cfg.optim_cfg, cfg.scheduler_cfg, params["task_cfg"], normalization_stats=normalization_stats)
    elif cfg.model_cfg.name.lower() == 'unet':
        # Pass task_cfg and normalization_stats to UNet model
        model = UNetLightning(cfg.model_cfg, cfg.optim_cfg, cfg.scheduler_cfg, params["task_cfg"], normalization_stats=normalization_stats)
    elif cfg.model_cfg.name.lower() == 'ffno':
        # Pass task_cfg and normalization_stats to FFNO model
        model = FFNOLightning(cfg.model_cfg, cfg.optim_cfg, cfg.scheduler_cfg, params["task_cfg"], normalization_stats=normalization_stats)
    elif cfg.model_cfg.name.lower() == 'flow_matching':
        model = ConditionalFlowMatchingLightning(cfg.model_cfg, cfg.optim_cfg, cfg.scheduler_cfg, params["task_cfg"], normalization_stats=normalization_stats, norm_mode=norm_mode)
    elif cfg.model_cfg.name.lower() == 'flow_matching_history':
        model = ConditionalFlowMatchingHistoryLightning(cfg.model_cfg, cfg.optim_cfg, cfg.scheduler_cfg, params["task_cfg"], normalization_stats=normalization_stats, norm_mode=norm_mode)
    elif cfg.model_cfg.name.lower() == 'flow_matching_jit':
        model = ConditionalFlowMatchingJiTLightning(cfg.model_cfg, cfg.optim_cfg, cfg.scheduler_cfg, params["task_cfg"], normalization_stats=normalization_stats, norm_mode=norm_mode)
    elif cfg.model_cfg.name.lower() == 've_sde':
        # Pass task_cfg and normalization_stats to VE-SDE model
        model = ScoreBasedVESDELightning(cfg.model_cfg, cfg.optim_cfg, cfg.scheduler_cfg, params["task_cfg"], normalization_stats=normalization_stats)
    elif cfg.model_cfg.name.lower() == 'flow_matching_ar':
        # Pass task_cfg and normalization_stats to autoregressive flow matching model
        model = ConditionalFlowMatchingARLightning(cfg.model_cfg, cfg.optim_cfg, cfg.scheduler_cfg, params["task_cfg"], normalization_stats=normalization_stats)
    elif cfg.model_cfg.name.lower() == 'flow_matching_ar_bootstrap':
        # Pass task_cfg and normalization_stats to bootstrap AR flow matching model
        model = ConditionalFlowMatchingARBootstrapLightning(cfg.model_cfg, cfg.optim_cfg, cfg.scheduler_cfg, params["task_cfg"], normalization_stats=normalization_stats)
    elif cfg.model_cfg.name.lower() == 'unet_ar':
        # Pass task_cfg and normalization_stats to autoregressive UNet model
        model = UNetARLightning(cfg.model_cfg, cfg.optim_cfg, cfg.scheduler_cfg, params["task_cfg"], normalization_stats=normalization_stats)
    elif cfg.model_cfg.name.lower() == 'edm':
        # Add downsample_factor to model_cfg (from data_cfg, not normalization_stats)
        # This ensures the model gets downsample_factor from the data configuration
        # Create a mutable copy of model_cfg and add downsample_factor
        model_cfg_dict = OmegaConf.to_container(cfg.model_cfg, resolve=True)
        model_cfg_dict['downsample_factor'] = downsample_factor
        model_cfg_updated = OmegaConf.create(model_cfg_dict)
        # Pass task_cfg and normalization_stats to EDM model (EDM-style diffusion baseline)
        model = EDMLightning(model_cfg_updated, cfg.optim_cfg, cfg.scheduler_cfg, params["task_cfg"], normalization_stats=normalization_stats)
    elif cfg.model_cfg.name.lower() == 'edm_ar_bootstrap':
        model_cfg_dict = OmegaConf.to_container(cfg.model_cfg, resolve=True)
        model_cfg_dict['downsample_factor'] = downsample_factor
        model_cfg_updated = OmegaConf.create(model_cfg_dict)
        model = EDMARBootstrapLightning(model_cfg_updated, cfg.optim_cfg, cfg.scheduler_cfg, params["task_cfg"], normalization_stats=normalization_stats)
    elif cfg.model_cfg.name.lower() == 'diffusionpde':
        model_cfg_dict = OmegaConf.to_container(cfg.model_cfg, resolve=True)
        model_cfg_dict['downsample_factor'] = downsample_factor
        model_cfg_updated = OmegaConf.create(model_cfg_dict)
        model = DiffusionPDELightning(model_cfg_updated, cfg.optim_cfg, cfg.scheduler_cfg, params["task_cfg"], normalization_stats=normalization_stats)
    else:
        raise ValueError(f"Unknown model: {cfg.model_cfg.name}")

    # Setup callbacks
    callbacks = [
        ModelSummary(max_depth=2),
        PreemptionCheckpointCallback(preempt_ckpt_path),
        ModelCheckpoint(
            dirpath=os.path.join(params["log_dir"], "checkpoints"),
            filename="{epoch:02d}-{step:06d}",
            monitor=params["checkpoint_monitor"],
            mode=params["checkpoint_mode"],
            save_top_k=params["save_top_k"],
            save_last=params["save_last"],
            save_on_train_epoch_end=True,
            verbose=True if is_leader_process() else False,
        )
    ]

    # Initialize trainer
    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        devices=params["devices"],
        num_nodes=params["nodes"],
        logger=loggers,
        callbacks=callbacks,
        plugins=[SLURMEnvironment(auto_requeue=False)],
        accelerator="auto",
        strategy="auto",
        # limit_train_batches=50,
        # Set validation batches to only 1% for faster validation
        limit_val_batches=0.1,
    )

    # Start training
    if params["checkpoint_path"] is not None:
        if is_leader_process():
            print(f"Starting training from checkpoint: {params['checkpoint_path']}")
        trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=params["checkpoint_path"])
    else:
        if is_leader_process():
            print("Starting training from scratch")
        trainer.fit(model, train_dataloader, val_dataloader)

    # Clean up wandb
    if params["use_wandb"] and is_leader_process():
        wandb.finish()

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()

