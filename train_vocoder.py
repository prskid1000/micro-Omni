
"""
Train HiFi-GAN neural vocoder for high-quality speech synthesis.

This script trains a HiFi-GAN vocoder to convert mel spectrograms to audio waveforms.
Uses adversarial training with Multi-Period Discriminator (MPD) and Multi-Scale Discriminator (MSD).
"""

import argparse
import json
import os
import csv
import pickle
import torch
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import torchaudio
from omni.codec import HiFiGANVocoder, MultiPeriodDiscriminator, MultiScaleDiscriminator
from omni.training_utils import set_seed, get_lr_scheduler, clip_gradients, SimpleLogger, cleanup_old_checkpoints
from tqdm import tqdm


def collate_mel_audio_fn(batch):
    """Collate function for mel spectrograms and audio pairs"""
    mels, audios = zip(*batch)
    
    # Pad mel spectrograms
    max_mel_len = max(m.shape[0] for m in mels)
    n_mels = mels[0].shape[1]
    padded_mels = []
    for m in mels:
        pad_len = max_mel_len - m.shape[0]
        if pad_len > 0:
            m = torch.cat([m, torch.zeros(pad_len, n_mels)], dim=0)
        padded_mels.append(m)
    
    # Pad audio waveforms
    max_audio_len = max(a.shape[0] for a in audios)
    padded_audios = []
    for a in audios:
        pad_len = max_audio_len - a.shape[0]
        if pad_len > 0:
            a = torch.cat([a, torch.zeros(pad_len)], dim=0)
        padded_audios.append(a)
    
    return torch.stack(padded_mels), torch.stack(padded_audios)


class VocoderDataset(Dataset):
    """Dataset for vocoder training - loads audio and computes mel spectrograms"""
    def __init__(self, csv_path, sr=16000, n_mels=128, n_fft=1024, hop_length=256, cfg=None):
        self.csv_path = csv_path
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Limit audio length to save memory (default: ~0.5 seconds at 16kHz)
        # Set to None to use full audio length
        self.max_audio_length = cfg.get("max_audio_length", None) if cfg else None
        
        # Mel spectrogram transform (matches vocoder input)
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            n_mels=n_mels,
            fmin=0.0,
            fmax=sr / 2.0
        )
        
        # Build or load cached row offset index (speeds up resuming)
        offset_cache_path = f"{csv_path}.row_offsets.pkl"
        file_mtime = os.path.getmtime(csv_path)
        cache_valid = False
        
        if os.path.exists(offset_cache_path):
            try:
                with open(offset_cache_path, 'rb') as cache_file:
                    cached_data = pickle.load(cache_file)
                    cached_mtime = cached_data.get('mtime', 0)
                    cached_offsets = cached_data.get('offsets', [])
                    cached_fieldnames = cached_data.get('fieldnames', None)
                    
                    # Check if cache is valid (file hasn't changed)
                    if abs(cached_mtime - file_mtime) < 1.0:  # Within 1 second tolerance
                        self.row_offsets = cached_offsets
                        self.fieldnames = cached_fieldnames
                        cache_valid = True
            except Exception:
                pass  # Cache invalid, will rebuild
        
        if not cache_valid:
            # Build row offset index for efficient random access
            self.row_offsets = []
            self.fieldnames = None
            with open(csv_path, 'rb') as f:
                header_line = f.readline()
                if not header_line:
                    raise ValueError("CSV file is empty")
                self.fieldnames = header_line.decode('utf-8').strip().split(',')
                offset = f.tell()
                while True:
                    line_start = offset
                    line_bytes = f.readline()
                    if not line_bytes:
                        break
                    try:
                        decoded = line_bytes.decode('utf-8').strip()
                        if decoded:
                            self.row_offsets.append(line_start)
                    except UnicodeDecodeError:
                        pass
                    offset += len(line_bytes)
            
            # Cache the offset index for future use
            try:
                with open(offset_cache_path, 'wb') as cache_file:
                    pickle.dump({'mtime': file_mtime, 'offsets': self.row_offsets, 'fieldnames': self.fieldnames}, cache_file)
            except Exception:
                pass  # Cache write failed, but continue anyway
    
    def __len__(self):
        return len(self.row_offsets)
    
    def __getitem__(self, i):
        """Load audio file and return mel spectrogram + audio waveform"""
        # Read specific row using file offset
        with open(self.csv_path, 'rb') as f:
            f.seek(self.row_offsets[i])
            line_bytes = f.readline()
            line = line_bytes.decode('utf-8').strip()
        
        # Parse CSV row
        import io
        reader = csv.DictReader(io.StringIO(line), fieldnames=self.fieldnames)
        row = next(reader)
        
        # Get audio path (works with both ASR and TTS CSV formats)
        if "wav" in row:
            path = row["wav"]
        elif "audio" in row:
            path = row["audio"]
        else:
            raise ValueError(f"CSV must have 'wav' or 'audio' column. Found: {self.fieldnames}")
        
        # Load audio
        audio, sr = torchaudio.load(path)
        if sr != self.sr:
            # Resample if needed
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            audio = resampler(audio)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        audio = audio.squeeze(0)  # (T,)
        
        # Truncate audio if too long (to save memory)
        if self.max_audio_length is not None and audio.shape[0] > self.max_audio_length:
            # Random crop for data augmentation
            if self.max_audio_length < audio.shape[0]:
                start_idx = torch.randint(0, audio.shape[0] - self.max_audio_length + 1, (1,)).item()
                audio = audio[start_idx:start_idx + self.max_audio_length]
            else:
                audio = audio[:self.max_audio_length]
        
        # Compute mel spectrogram
        mel = self.melspec(audio.unsqueeze(0))[0].T  # (T_mel, n_mels)
        
        # Normalize mel to [0, 1] range (for training stability)
        mel_min = mel.min()
        mel_max = mel.max()
        if mel_max > mel_min + 1e-6:
            mel = (mel - mel_min) / (mel_max - mel_min + 1e-8)
        
        return mel, audio


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
    
    ds = VocoderDataset(csv_path, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, cfg=cfg)
    dl = DataLoader(
        ds, 
        batch_size=cfg.get("batch_size", 2), 
        shuffle=True, 
        num_workers=cfg.get("num_workers", 1),  # Reduced for 12GB VRAM
        drop_last=cfg.get("drop_last", True),
        collate_fn=collate_mel_audio_fn,
        pin_memory=True,  # Faster GPU transfer
        prefetch_factor=2,  # Prefetch batches
        persistent_workers=True if cfg.get("num_workers", 1) > 0 else False  # Keep workers alive
    )
    
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
    total_size = len(ds)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [train_size, val_size], 
        generator=torch.Generator().manual_seed(seed)
    )
    train_dl = DataLoader(
        train_ds, 
        batch_size=cfg.get("batch_size", 4), 
        shuffle=True, 
        num_workers=cfg.get("num_workers", 2), 
        drop_last=True,
        collate_fn=collate_mel_audio_fn
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
    start_epoch = 0
    resume_from = None
    if os.path.exists(cfg["save_dir"]):
        checkpoint_files = [f for f in os.listdir(cfg["save_dir"]) if f.startswith("vocoder_step_") and f.endswith(".pt")]
        if checkpoint_files:
            step_numbers = []
            for f in checkpoint_files:
                try:
                    step_num = int(f.replace("vocoder_step_", "").replace(".pt", ""))
                    step_numbers.append((step_num, f))
                except:
                    continue
            if step_numbers:
                step_numbers.sort(key=lambda x: x[0], reverse=True)
                resume_from = os.path.join(cfg["save_dir"], step_numbers[0][1])
                step = step_numbers[0][0]
                logger.info(f"Found checkpoint at step {step}, resuming from: {resume_from}")
                
                checkpoint = torch.load(resume_from, map_location=device)
                if "generator" in checkpoint:
                    generator.load_state_dict(checkpoint["generator"])
                if "mpd" in checkpoint:
                    mpd.load_state_dict(checkpoint["mpd"])
                if "msd" in checkpoint:
                    msd.load_state_dict(checkpoint["msd"])
                if "opt_g" in checkpoint:
                    opt_g.load_state_dict(checkpoint["opt_g"])
                if "opt_d" in checkpoint:
                    opt_d.load_state_dict(checkpoint["opt_d"])
                if "scheduler_g" in checkpoint:
                    scheduler_g.load_state_dict(checkpoint["scheduler_g"])
                if "scheduler_d" in checkpoint:
                    scheduler_d.load_state_dict(checkpoint["scheduler_d"])
                if "step" in checkpoint:
                    step = checkpoint["step"]
                logger.info(f"Resumed from step {step}")
    
    logger.training_start(max_steps, train_size, val_size)
    
    steps_per_epoch = len(train_dl)
    initial_step = step
    if step > 0:
        start_epoch = step // steps_per_epoch
    
    max_epochs = cfg.get("max_epochs", 9999)
    print_freq = cfg.get("print_freq", 50)
    checkpoint_freq = cfg.get("checkpoint_freq", 1000)
    val_freq = cfg.get("val_freq", 500)
    
    generator.train()
    mpd.train()
    msd.train()
    
    for epoch in range(start_epoch, max_epochs):
        for batch_idx, (mel, audio_real) in enumerate(tqdm(train_dl, desc=f"epoch{epoch}")):
            # Skip batches if resuming
            if epoch == start_epoch and initial_step > 0:
                current_batch_step = epoch * steps_per_epoch + batch_idx
                if current_batch_step < initial_step:
                    continue
            
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
                    clip_gradients(mpd, cfg.get("max_grad_norm", 1.0))
                    clip_gradients(msd, cfg.get("max_grad_norm", 1.0))
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
                    clip_gradients(mpd, cfg.get("max_grad_norm", 1.0))
                    clip_gradients(msd, cfg.get("max_grad_norm", 1.0))
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
                    clip_gradients(generator, cfg.get("max_grad_norm", 1.0))
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
                    clip_gradients(generator, cfg.get("max_grad_norm", 1.0))
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

