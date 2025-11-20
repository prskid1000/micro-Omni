
"""
Train HiFi-GAN neural vocoder for high-quality speech synthesis.

This script trains a HiFi-GAN vocoder to convert mel spectrograms to audio waveforms.
Uses adversarial training with Multi-Period Discriminator (MPD) and Multi-Scale Discriminator (MSD).
"""

import argparse
import json
import os
import torch
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from omni.codec import HiFiGANVocoder, MultiPeriodDiscriminator, MultiScaleDiscriminator
from omni.utils import (
    set_seed, get_lr_scheduler, clip_gradients, SimpleLogger, cleanup_old_checkpoints, VocoderDataset,
    load_checkpoint, setup_resume_data_loading, calculate_resume_position,
    ValidationSkipSamplesContext, check_gradient_explosion, collate_mel_audio_fn
)
from tqdm import tqdm

def discriminator_loss(real_outputs, fake_outputs):
    """Compute discriminator loss (adversarial loss)"""
    loss = 0.0
    count = 0
    # Handle nested lists (from MPD/MSD)
    for real_outs, fake_outs in zip(real_outputs, fake_outputs):
        if isinstance(real_outs, list):
            # Multiple outputs from one discriminator
            for real_out, fake_out in zip(real_outs, fake_outs):
                # Real should be 1, fake should be 0
                real_loss = torch.mean((real_out - 1.0) ** 2)
                fake_loss = torch.mean(fake_out ** 2)
                loss += (real_loss + fake_loss) / 2.0
                count += 1
        else:
            # Single output
            real_loss = torch.mean((real_outs - 1.0) ** 2)
            fake_loss = torch.mean(fake_outs ** 2)
            loss += (real_loss + fake_loss) / 2.0
            count += 1
    return loss / max(count, 1)


def generator_loss(fake_outputs):
    """Compute generator adversarial loss"""
    loss = 0.0
    count = 0
    # Handle nested lists (from MPD/MSD)
    for fake_outs in fake_outputs:
        if isinstance(fake_outs, list):
            # Multiple outputs from one discriminator
            for fake_out in fake_outs:
                # Generator wants discriminator to output 1 for fake
                loss += torch.mean((fake_out - 1.0) ** 2)
                count += 1
        else:
            # Single output
            loss += torch.mean((fake_outs - 1.0) ** 2)
            count += 1
    return loss / max(count, 1)


def feature_matching_loss(real_feats, fake_feats):
    """Compute feature matching loss for generator"""
    loss = 0.0
    count = 0
    # Handle nested lists (from MPD/MSD)
    for real_feat_list, fake_feat_list in zip(real_feats, fake_feats):
        if isinstance(real_feat_list, list):
            for r, f in zip(real_feat_list, fake_feat_list):
                loss += torch.mean(torch.abs(r - f))
                count += 1
        else:
            loss += torch.mean(torch.abs(real_feat_list - fake_feat_list))
            count += 1
    return loss / max(count, 1)


def mel_loss(mel_real, mel_fake):
    """Compute mel spectrogram loss"""
    return torch.mean(torch.abs(mel_real - mel_fake))


def main(cfg):
    # Set random seed
    seed = cfg.get("seed", 42)
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["save_dir"], exist_ok=True)
    
    sr = cfg.get("sample_rate", 16000)
    n_mels = cfg.get("n_mels", 128)
    n_fft = cfg.get("n_fft", 1024)
    hop_length = cfg.get("hop_length", 256)
    
    # Load dataset
    csv_path = cfg.get("train_csv", "data/audio/production_tts.csv")
    if not os.path.exists(csv_path):
        # Try ASR CSV as fallback
        csv_path = cfg.get("train_csv", "data/audio/production_asr.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Audio CSV not found. Expected: {csv_path}")
    
    
    # Initialize models
    generator = HiFiGANVocoder(
        sample_rate=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    ).to(device)
    
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    
    # Optimizers
    opt_g = torch.optim.AdamW(
        generator.parameters(),
        lr=cfg.get("lr_g", 2e-4),
        betas=(0.8, 0.99),
        weight_decay=cfg.get("wd", 1e-6)
    )
    opt_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=cfg.get("lr_d", 2e-4),
        betas=(0.8, 0.99),
        weight_decay=cfg.get("wd", 1e-6)
    )
    
    # Learning rate schedulers
    warmup_steps = cfg.get("warmup_steps", 1000)
    max_steps = cfg.get("max_steps", 100000)
    scheduler_g = get_lr_scheduler(opt_g, warmup_steps, max_steps)
    scheduler_d = get_lr_scheduler(opt_d, warmup_steps, max_steps)
    
    # Mixed precision (FP16) - saves ~50% memory, 2x faster
    use_amp = cfg.get("use_amp", True) and device == "cuda"
    scaler_g = GradScaler('cuda') if use_amp else None
    scaler_d = GradScaler('cuda') if use_amp else None
    if use_amp:
        print("✓ Mixed precision training (FP16) enabled - saves ~50% memory")
    
    # Gradient accumulation (important for memory efficiency)
    # Simulates larger batch size without OOM
    accumulation_steps = cfg.get("gradient_accumulation_steps", 4)
    effective_batch_size = cfg.get("batch_size", 2) * accumulation_steps
    if accumulation_steps > 1:
        print(f"✓ Gradient accumulation: {accumulation_steps} steps (effective batch size: {effective_batch_size})")
    
    # Memory optimizations summary
    print(f"\n📊 Memory Optimizations for 12GB VRAM:")
    print(f"  • Batch size: {cfg.get('batch_size', 2)}")
    print(f"  • Effective batch size: {effective_batch_size} (with gradient accumulation)")
    print(f"  • Audio length limit: {cfg.get('max_audio_length', 8192)} samples (~{cfg.get('max_audio_length', 8192)/sr:.2f}s)")
    print(f"  • DataLoader workers: {cfg.get('num_workers', 1)}")
    print(f"  • Mixed precision: {'Enabled' if use_amp else 'Disabled'}")
    print()
    
    # Loss weights
    lambda_mel = cfg.get("lambda_mel", 45.0)
    lambda_fm = cfg.get("lambda_fm", 2.0)
    lambda_adv = cfg.get("lambda_adv", 1.0)
    
    # Validation split
    val_split = cfg.get("val_split", 0.1)
    
    train_ds = VocoderDataset(
        csv_path, 
        sr=sr, 
        n_mels=n_mels, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        cfg=cfg,
        shuffle_buffer_size=cfg.get("shuffle_buffer_size", 10000),
        seed=seed,
        skip_samples=0
    )
    train_ds._val_split = val_split
    train_ds._val_mode = False  # Training mode
    
    val_ds = VocoderDataset(
        csv_path, 
        sr=sr, 
        n_mels=n_mels, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        cfg=cfg,
        shuffle_buffer_size=0,  # No shuffling for validation
        seed=seed,  # Same seed for consistent hash-based split
        skip_samples=0
    )
    val_ds._val_split = val_split
    val_ds._val_mode = True  # Validation mode
    
    # Approximate sizes for logging (will count if needed)
    try:
        total_size = train_ds.get_length()
        train_size = int(total_size * (1 - val_split))
        val_size = total_size - train_size
    except:
        train_size = val_size = None  # Unknown size
    
    # Note: shuffle=False for IterableDataset (shuffling handled internally)
    train_dl = DataLoader(
        train_ds, 
        batch_size=cfg.get("batch_size", 2), 
        shuffle=False, 
        num_workers=cfg.get("num_workers", 1),  # Reduced for 12GB VRAM
        drop_last=True,
        collate_fn=collate_mel_audio_fn,
        pin_memory=True,  # Faster GPU transfer
        prefetch_factor=2,  # Prefetch batches
        persistent_workers=True if cfg.get("num_workers", 1) > 0 else False  # Keep workers alive
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=cfg.get("batch_size", 4), 
        shuffle=False, 
        num_workers=cfg.get("num_workers", 2), 
        drop_last=False,
        collate_fn=collate_mel_audio_fn
    )
    
    # Logger
    logger = SimpleLogger("HiFi-GAN Vocoder")
    
    # Resume from checkpoint
    step = 0
    step, resume_from = load_checkpoint(
        cfg["save_dir"], 
        "vocoder_step_", 
        device, 
        logger,
        state_dict_loaders={
            "generator": (generator, generator.load_state_dict),
            "mpd": (mpd, mpd.load_state_dict),
            "msd": (msd, msd.load_state_dict),
            "opt_g": (opt_g, opt_g.load_state_dict),
            "opt_d": (opt_d, opt_d.load_state_dict),
            "scheduler_g": (scheduler_g, scheduler_g.load_state_dict),
            "scheduler_d": (scheduler_d, scheduler_d.load_state_dict)
        }
    )
    
    # Update skip_samples for dataset if resuming
    batch_size = cfg.get("batch_size", 4)
    new_train_dl = setup_resume_data_loading(
        train_ds, step, batch_size, logger,
        train_dl_kwargs={
            "num_workers": cfg.get("num_workers", 2),
            "drop_last": True,
            "collate_fn": collate_mel_audio_fn
        }
    )
    if new_train_dl is not None:
        train_dl = new_train_dl
    
    logger.training_start(max_steps, train_size, val_size)
    
    # Calculate steps per epoch and determine starting epoch/position
    # For IterableDataset, we can't use len() directly, so calculate from dataset size
    batch_size = cfg.get("batch_size", 4)
    drop_last = True  # Vocoder uses drop_last=True
    if train_size is not None:
        steps_per_epoch = train_size // batch_size
        if not drop_last and train_size % batch_size != 0:
            steps_per_epoch += 1
    else:
        # Fallback: use a large number if size is unknown (for progress bar)
        # The actual training will work fine, just progress bar won't be accurate
        steps_per_epoch = 1000000  # Large placeholder
    initial_step = step
    start_epoch, start_batch_idx = calculate_resume_position(step, steps_per_epoch)
    if step > 0:
        logger.info(f"Resuming from step {step} (epoch {start_epoch}, batch {start_batch_idx}/{steps_per_epoch})")
    
    max_epochs = cfg.get("max_epochs", 9999)
    print_freq = cfg.get("print_freq", 50)
    checkpoint_freq = cfg.get("checkpoint_freq", 1000)
    val_freq = cfg.get("val_freq", 500)
    
    generator.train()
    mpd.train()
    msd.train()
    
    for epoch in range(start_epoch, max_epochs):
        # Recreate DataLoader for each epoch since IterableDatasets are exhausted after one iteration
        # skip_samples is automatically reset to 0 by the dataset after first iteration
        if epoch > start_epoch:
            train_dl = DataLoader(
                train_ds, 
                batch_size=cfg.get("batch_size", 2), 
                shuffle=False, 
                num_workers=cfg.get("num_workers", 1),
                drop_last=True,
                collate_fn=collate_mel_audio_fn,
                pin_memory=True
            )
        
        # Create progress bar with correct starting position when resuming mid-epoch
        remaining_epochs = max_epochs - epoch - 1
        pbar_desc = f"epoch{epoch}/{max_epochs-1} (remaining:{remaining_epochs}) step{step}"
        if epoch == start_epoch and start_batch_idx > 0:
            pbar = tqdm(train_dl, desc=pbar_desc, initial=start_batch_idx, total=steps_per_epoch)
        else:
            pbar = tqdm(train_dl, desc=pbar_desc, total=steps_per_epoch)
        
        # Start enumeration from the correct position when resuming mid-epoch
        enum_start = start_batch_idx if (epoch == start_epoch and start_batch_idx > 0) else 0
        for batch_idx, (mel, audio_real) in enumerate(pbar, start=enum_start):
            # Skip batches if resuming mid-epoch
            # batch_idx already represents the position in the epoch when enum_start > 0
            if epoch == start_epoch and initial_step > 0:
                current_batch_step = epoch * steps_per_epoch + batch_idx
                if current_batch_step < initial_step:
                    continue
            
            # Update progress bar description
            remaining_epochs = max_epochs - epoch - 1
            pbar.set_description(f"epoch{epoch}/{max_epochs-1} (remaining:{remaining_epochs}) step{step} batch{batch_idx}")
            
            mel = mel.to(device)  # (B, T_mel, n_mels)
            audio_real = audio_real.to(device)  # (B, T_audio)
            
            # Reshape mel for generator: (B, n_mels, T_mel)
            mel_input = mel.transpose(1, 2)  # (B, n_mels, T_mel)
            
            # ========== Train Discriminators ==========
            opt_d.zero_grad()
            
            # Generate fake audio
            with torch.no_grad():
                audio_fake = generator(mel_input)  # (B, T_audio)
            
            # Ensure same length
            min_len = min(audio_real.shape[1], audio_fake.shape[1])
            audio_real = audio_real[:, :min_len]
            audio_fake = audio_fake[:, :min_len]
            
            # Add channel dimension for discriminators: (B, 1, T)
            audio_real_d = audio_real.unsqueeze(1)
            audio_fake_d = audio_fake.unsqueeze(1)
            
            if use_amp:
                with autocast(device_type='cuda'):
                    # MPD - returns list of outputs and features
                    mpd_real_out, mpd_real_feats = mpd(audio_real_d)
                    mpd_fake_out, mpd_fake_feats = mpd(audio_fake_d.detach())
                    # MPD returns list of outputs (one per period)
                    loss_mpd = 0.0
                    for real_out, fake_out in zip(mpd_real_out, mpd_fake_out):
                        real_loss = torch.mean((real_out - 1.0) ** 2)
                        fake_loss = torch.mean(fake_out ** 2)
                        loss_mpd += (real_loss + fake_loss) / 2.0
                    loss_mpd = loss_mpd / len(mpd_real_out)
                    
                    # MSD - returns list of outputs and features
                    msd_real_out, msd_real_feats = msd(audio_real_d)
                    msd_fake_out, msd_fake_feats = msd(audio_fake_d.detach())
                    # MSD returns list of outputs (one per scale)
                    loss_msd = 0.0
                    for real_out, fake_out in zip(msd_real_out, msd_fake_out):
                        real_loss = torch.mean((real_out - 1.0) ** 2)
                        fake_loss = torch.mean(fake_out ** 2)
                        loss_msd += (real_loss + fake_loss) / 2.0
                    loss_msd = loss_msd / len(msd_real_out)
                    
                    loss_d = loss_mpd + loss_msd
                
                scaler_d.scale(loss_d / accumulation_steps).backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler_d.unscale_(opt_d)
                    max_grad_norm = cfg.get("max_grad_norm", 1.0)
                    
                    # Gradient clipping first, then check if still too high
                    try:
                        grad_norm_mpd_before = clip_gradients(mpd, max_grad_norm)
                        grad_norm_msd_before = clip_gradients(msd, max_grad_norm)
                        
                        # Check for gradient explosion AFTER clipping
                        explosion_threshold = max(100.0, max_grad_norm * 10)
                        grad_norm_mpd_after, is_exploded_mpd = check_gradient_explosion(mpd, max_grad_norm=explosion_threshold, raise_on_error=False)
                        grad_norm_msd_after, is_exploded_msd = check_gradient_explosion(msd, max_grad_norm=explosion_threshold, raise_on_error=False)
                        
                        if is_exploded_mpd or is_exploded_msd:
                            logger.error(f"Step {step}: Discriminator gradient explosion after clipping (mpd: {grad_norm_mpd_before:.2f}->{grad_norm_mpd_after:.2f}, msd: {grad_norm_msd_before:.2f}->{grad_norm_msd_after:.2f}). Skipping this batch.")
                            opt_d.zero_grad()
                            scaler_d.update()
                            continue
                    except RuntimeError as e:
                        logger.error(f"Step {step}: {e}")
                        opt_d.zero_grad()
                        scaler_d.update()
                        continue
                    
                    scaler_d.step(opt_d)
                    scaler_d.update()
                    scheduler_d.step()
                    opt_d.zero_grad()
            else:
                # MPD - returns list of outputs and features
                mpd_real_out, mpd_real_feats = mpd(audio_real_d)
                mpd_fake_out, mpd_fake_feats = mpd(audio_fake_d.detach())
                # MPD returns list of outputs (one per period)
                loss_mpd = 0.0
                for real_out, fake_out in zip(mpd_real_out, mpd_fake_out):
                    real_loss = torch.mean((real_out - 1.0) ** 2)
                    fake_loss = torch.mean(fake_out ** 2)
                    loss_mpd += (real_loss + fake_loss) / 2.0
                loss_mpd = loss_mpd / len(mpd_real_out)
                
                # MSD - returns list of outputs and features
                msd_real_out, msd_real_feats = msd(audio_real_d)
                msd_fake_out, msd_fake_feats = msd(audio_fake_d.detach())
                # MSD returns list of outputs (one per scale)
                loss_msd = 0.0
                for real_out, fake_out in zip(msd_real_out, msd_fake_out):
                    real_loss = torch.mean((real_out - 1.0) ** 2)
                    fake_loss = torch.mean(fake_out ** 2)
                    loss_msd += (real_loss + fake_loss) / 2.0
                loss_msd = loss_msd / len(msd_real_out)
                
                loss_d = loss_mpd + loss_msd
                (loss_d / accumulation_steps).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    max_grad_norm = cfg.get("max_grad_norm", 1.0)
                    
                    # Gradient clipping first, then check if still too high
                    try:
                        grad_norm_mpd_before = clip_gradients(mpd, max_grad_norm)
                        grad_norm_msd_before = clip_gradients(msd, max_grad_norm)
                        
                        # Check for gradient explosion AFTER clipping
                        explosion_threshold = max(100.0, max_grad_norm * 10)
                        grad_norm_mpd_after, is_exploded_mpd = check_gradient_explosion(mpd, max_grad_norm=explosion_threshold, raise_on_error=False)
                        grad_norm_msd_after, is_exploded_msd = check_gradient_explosion(msd, max_grad_norm=explosion_threshold, raise_on_error=False)
                        
                        if is_exploded_mpd or is_exploded_msd:
                            logger.error(f"Step {step}: Discriminator gradient explosion after clipping (mpd: {grad_norm_mpd_before:.2f}->{grad_norm_mpd_after:.2f}, msd: {grad_norm_msd_before:.2f}->{grad_norm_msd_after:.2f}). Skipping this batch.")
                            opt_d.zero_grad()
                            continue
                    except RuntimeError as e:
                        logger.error(f"Step {step}: {e}")
                        opt_d.zero_grad()
                        continue
                    
                    opt_d.step()
                    scheduler_d.step()
                    opt_d.zero_grad()
            
            # ========== Train Generator ==========
            opt_g.zero_grad()
            
            # Generate fake audio
            audio_fake = generator(mel_input)  # (B, T_audio)
            
            # Ensure same length
            min_len = min(audio_real.shape[1], audio_fake.shape[1])
            audio_real = audio_real[:, :min_len]
            audio_fake = audio_fake[:, :min_len]
            audio_fake_d = audio_fake.unsqueeze(1)
            
            # Compute mel spectrogram of generated audio
            melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                n_mels=n_mels,
                fmin=0.0,
                fmax=sr / 2.0
            ).to(device)
            mel_fake = melspec(audio_fake.unsqueeze(1))[0].T  # (T_mel, n_mels)
            mel_real = mel[:, :mel_fake.shape[0], :]  # Match length
            
            if use_amp:
                with autocast(device_type='cuda'):
                    # Adversarial losses
                    mpd_fake_out_list, mpd_fake_feats = mpd(audio_fake_d)
                    # Extract final outputs from each period discriminator
                    loss_adv_mpd = 0.0
                    for fake_out in mpd_fake_out_list:
                        loss_adv_mpd += torch.mean((fake_out - 1.0) ** 2)
                    loss_adv_mpd = loss_adv_mpd / len(mpd_fake_out_list)
                    
                    msd_fake_out_list, msd_fake_feats = msd(audio_fake_d)
                    # Extract final outputs from each scale discriminator
                    loss_adv_msd = 0.0
                    for fake_out in msd_fake_out_list:
                        loss_adv_msd += torch.mean((fake_out - 1.0) ** 2)
                    loss_adv_msd = loss_adv_msd / len(msd_fake_out_list)
                    
                    # Feature matching losses
                    _, mpd_real_feats = mpd(audio_real_d[:, :, :min_len])
                    loss_fm_mpd = 0.0
                    count_fm = 0
                    for real_feat_list, fake_feat_list in zip(mpd_real_feats, mpd_fake_feats):
                        for r, f in zip(real_feat_list, fake_feat_list):
                            loss_fm_mpd += torch.mean(torch.abs(r - f))
                            count_fm += 1
                    loss_fm_mpd = loss_fm_mpd / max(count_fm, 1)
                    
                    _, msd_real_feats = msd(audio_real_d[:, :, :min_len])
                    loss_fm_msd = 0.0
                    count_fm = 0
                    for real_feat_list, fake_feat_list in zip(msd_real_feats, msd_fake_feats):
                        for r, f in zip(real_feat_list, fake_feat_list):
                            loss_fm_msd += torch.mean(torch.abs(r - f))
                            count_fm += 1
                    loss_fm_msd = loss_fm_msd / max(count_fm, 1)
                    
                    # Mel spectrogram loss
                    loss_mel = mel_loss(mel_real, mel_fake)
                    
                    # Total generator loss
                    loss_g = (
                        lambda_adv * (loss_adv_mpd + loss_adv_msd) +
                        lambda_fm * (loss_fm_mpd + loss_fm_msd) +
                        lambda_mel * loss_mel
                    )
                
                scaler_g.scale(loss_g / accumulation_steps).backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler_g.unscale_(opt_g)
                    max_grad_norm = cfg.get("max_grad_norm", 1.0)
                    
                    # Gradient clipping first, then check if still too high
                    try:
                        grad_norm_gen_before = clip_gradients(generator, max_grad_norm)
                        
                        # Check for gradient explosion AFTER clipping
                        explosion_threshold = max(100.0, max_grad_norm * 10)
                        grad_norm_gen_after, is_exploded_gen = check_gradient_explosion(generator, max_grad_norm=explosion_threshold, raise_on_error=False)
                        
                        if is_exploded_gen:
                            logger.error(f"Step {step}: Generator gradient explosion after clipping (norm: {grad_norm_gen_before:.2f}->{grad_norm_gen_after:.2f}). Skipping this batch.")
                            opt_g.zero_grad()
                            scaler_g.update()
                            continue
                    except RuntimeError as e:
                        logger.error(f"Step {step}: {e}")
                        opt_g.zero_grad()
                        scaler_g.update()
                        continue
                    
                    scaler_g.step(opt_g)
                    scaler_g.update()
                    scheduler_g.step()
                    opt_g.zero_grad()
            else:
                # Adversarial losses
                mpd_fake_out_list, mpd_fake_feats = mpd(audio_fake_d)
                # Extract final outputs from each period discriminator
                loss_adv_mpd = 0.0
                for fake_out in mpd_fake_out_list:
                    loss_adv_mpd += torch.mean((fake_out - 1.0) ** 2)
                loss_adv_mpd = loss_adv_mpd / len(mpd_fake_out_list)
                
                msd_fake_out_list, msd_fake_feats = msd(audio_fake_d)
                # Extract final outputs from each scale discriminator
                loss_adv_msd = 0.0
                for fake_out in msd_fake_out_list:
                    loss_adv_msd += torch.mean((fake_out - 1.0) ** 2)
                loss_adv_msd = loss_adv_msd / len(msd_fake_out_list)
                
                # Feature matching losses
                _, mpd_real_feats = mpd(audio_real_d[:, :, :min_len])
                loss_fm_mpd = 0.0
                count_fm = 0
                for real_feat_list, fake_feat_list in zip(mpd_real_feats, mpd_fake_feats):
                    for r, f in zip(real_feat_list, fake_feat_list):
                        loss_fm_mpd += torch.mean(torch.abs(r - f))
                        count_fm += 1
                loss_fm_mpd = loss_fm_mpd / max(count_fm, 1)
                
                _, msd_real_feats = msd(audio_real_d[:, :, :min_len])
                loss_fm_msd = 0.0
                count_fm = 0
                for real_feat_list, fake_feat_list in zip(msd_real_feats, msd_fake_feats):
                    for r, f in zip(real_feat_list, fake_feat_list):
                        loss_fm_msd += torch.mean(torch.abs(r - f))
                        count_fm += 1
                loss_fm_msd = loss_fm_msd / max(count_fm, 1)
                
                # Mel spectrogram loss
                loss_mel = mel_loss(mel_real, mel_fake)
                
                # Total generator loss
                loss_g = (
                    lambda_adv * (loss_adv_mpd + loss_adv_msd) +
                    lambda_fm * (loss_fm_mpd + loss_fm_msd) +
                    lambda_mel * loss_mel
                )
                
                (loss_g / accumulation_steps).backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    max_grad_norm = cfg.get("max_grad_norm", 1.0)
                    
                    # Gradient clipping first, then check if still too high
                    try:
                        grad_norm_gen_before = clip_gradients(generator, max_grad_norm)
                        
                        # Check for gradient explosion AFTER clipping
                        explosion_threshold = max(100.0, max_grad_norm * 10)
                        grad_norm_gen_after, is_exploded_gen = check_gradient_explosion(generator, max_grad_norm=explosion_threshold, raise_on_error=False)
                        
                        if is_exploded_gen:
                            logger.error(f"Step {step}: Generator gradient explosion after clipping (norm: {grad_norm_gen_before:.2f}->{grad_norm_gen_after:.2f}). Skipping this batch.")
                            opt_g.zero_grad()
                            continue
                    except RuntimeError as e:
                        logger.error(f"Step {step}: {e}")
                        opt_g.zero_grad()
                        continue
                    
                    opt_g.step()
                    scheduler_g.step()
                    opt_g.zero_grad()
            
            step += 1
            
            # Logging
            if step % print_freq == 0:
                logger.log(
                    step, 
                    {
                        "loss_g": loss_g.item(),
                        "loss_d": loss_d.item(),
                        "loss_mel": loss_mel.item(),
                        "loss_adv": (loss_adv_mpd + loss_adv_msd).item(),
                        "loss_fm": (loss_fm_mpd + loss_fm_msd).item(),
                        "lr_g": scheduler_g.get_last_lr()[0],
                        "lr_d": scheduler_d.get_last_lr()[0]
                    }
                )
            
            # Validation
            if step % val_freq == 0:
                with ValidationSkipSamplesContext(train_ds):
                    generator.eval()
                    mpd.eval()
                    msd.eval()
                    
                    val_loss_g = 0.0
                    val_loss_d = 0.0
                    val_samples = 0
                    
                    with torch.no_grad():
                        for val_mel, val_audio in val_dl:
                            if val_samples >= 100:  # Limit validation samples
                                break
                            
                            val_mel = val_mel.to(device)
                            val_audio = val_audio.to(device)
                            val_mel_input = val_mel.transpose(1, 2)
                            
                            # Generate fake audio
                            val_audio_fake = generator(val_mel_input)
                            min_len = min(val_audio.shape[1], val_audio_fake.shape[1])
                            val_audio = val_audio[:, :min_len]
                            val_audio_fake = val_audio_fake[:, :min_len]
                            
                            val_audio_real_d = val_audio.unsqueeze(1)
                            val_audio_fake_d = val_audio_fake.unsqueeze(1)
                            
                            # Discriminator loss
                            mpd_real_out, _ = mpd(val_audio_real_d)
                            mpd_fake_out, _ = mpd(val_audio_fake_d)
                            msd_real_out, _ = msd(val_audio_real_d)
                            msd_fake_out, _ = msd(val_audio_fake_d)
                            
                            # Compute MPD loss
                            loss_mpd = 0.0
                            for real_out, fake_out in zip(mpd_real_out, mpd_fake_out):
                                real_loss = torch.mean((real_out - 1.0) ** 2)
                                fake_loss = torch.mean(fake_out ** 2)
                                loss_mpd += (real_loss + fake_loss) / 2.0
                            val_loss_d += (loss_mpd / len(mpd_real_out)).item()
                            
                            # Compute MSD loss
                            loss_msd = 0.0
                            for real_out, fake_out in zip(msd_real_out, msd_fake_out):
                                real_loss = torch.mean((real_out - 1.0) ** 2)
                                fake_loss = torch.mean(fake_out ** 2)
                                loss_msd += (real_loss + fake_loss) / 2.0
                            val_loss_d += (loss_msd / len(msd_real_out)).item()
                            
                            # Generator loss (simplified)
                            val_loss_g += torch.mean((val_audio_fake - val_audio) ** 2).item()
                            
                            val_samples += 1
                    
                    val_loss_g /= val_samples
                    val_loss_d /= val_samples
                    logger.log(step, {"val_loss_g": val_loss_g, "val_loss_d": val_loss_d})
                    
                    generator.train()
                    mpd.train()
                    msd.train()
            
            # Checkpointing
            if step % checkpoint_freq == 0:
                checkpoint_path = os.path.join(cfg["save_dir"], f"vocoder_step_{step}.pt")
                torch.save({
                    "generator": generator.state_dict(),
                    "mpd": mpd.state_dict(),
                    "msd": msd.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "scheduler_g": scheduler_g.state_dict(),
                    "scheduler_d": scheduler_d.state_dict(),
                    "step": step,
                    "config": cfg
                }, checkpoint_path)
                
                # Save final checkpoint for inference
                final_path = os.path.join(cfg["save_dir"], "hifigan.pt")
                torch.save({
                    "generator": generator.state_dict(),
                    "config": cfg
                }, final_path)
                
                logger.info(f"Saved checkpoint: {checkpoint_path}")
                logger.info(f"Saved final checkpoint: {final_path}")
                
                # Cleanup old checkpoints
                cleanup_old_checkpoints(cfg["save_dir"], "vocoder_step_", keep_last=3)
            
            if step >= max_steps:
                logger.info(f"Reached max_steps ({max_steps}), stopping training")
                break
        
        if step >= max_steps:
            break
    
    # Save final model
    final_path = os.path.join(cfg["save_dir"], "hifigan.pt")
    torch.save({
        "generator": generator.state_dict(),
        "config": cfg
    }, final_path)
    logger.info(f"Training complete! Final model saved to: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HiFi-GAN neural vocoder")
    parser.add_argument("--config", type=str, default="configs/vocoder_tiny.json",
                       help="Path to config file")
    args = parser.parse_args()
    
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = json.load(f)
    else:
        # Default config
        cfg = {
            "save_dir": "checkpoints/vocoder_tiny",
            "train_csv": "data/audio/production_tts.csv",
            "sample_rate": 16000,
            "n_mels": 128,
            "n_fft": 1024,
            "hop_length": 256,
            "batch_size": 4,
            "num_workers": 2,
            "drop_last": True,
            "lr_g": 2e-4,
            "lr_d": 2e-4,
            "wd": 1e-6,
            "warmup_steps": 1000,
            "max_steps": 100000,
            "max_epochs": 9999,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "use_amp": True,
            "lambda_mel": 45.0,
            "lambda_fm": 2.0,
            "lambda_adv": 1.0,
            "val_split": 0.1,
            "print_freq": 50,
            "checkpoint_freq": 1000,
            "val_freq": 500,
            "seed": 42
        }
        print(f"Config file not found, using defaults. Creating: {args.config}")
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        with open(args.config, 'w') as f:
            json.dump(cfg, f, indent=2)
    
    main(cfg)

