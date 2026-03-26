"""
Complete test script to validate SFT (multimodal) stage accuracy.
Tests text-only, image+text, and audio+text inference.
Measures val loss, perplexity, and verifies multimodal feature integration.
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torchaudio  # Only used for transforms, not for loading audio
import json
import os
import csv
import argparse
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from omni.thinker import ThinkerLM
from omni.audio_encoder import AudioEncoderTiny
from omni.vision_encoder import ViTTiny
from omni.tokenizer import BPETokenizer
from omni.utils import find_checkpoint, strip_orig_mod, load_audio, enable_log_file, default_log_path

torch.set_float32_matmul_precision('high')


# ============================================================================
# Config Resolution (same pattern as other scripts: --config flag, fallback)
# ============================================================================

def resolve_config(args_config, checkpoint_dir):
    """
    Resolve config with fallback chain:
    1. Explicit --config flag
    2. Config embedded in checkpoint
    3. configs/<checkpoint_dir_name>.json
    """
    # 1. Explicit config file
    if args_config and os.path.exists(args_config):
        print(f"Loading config from: {args_config}")
        with open(args_config, 'r') as f:
            return json.load(f)

    # 2. Try loading from checkpoint (checked later during model load)
    # 3. Fallback to configs/<name>.json
    checkpoint_name = os.path.basename(checkpoint_dir)
    config_path = f"configs/{checkpoint_name}.json"
    if os.path.exists(config_path):
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            return json.load(f)

    raise FileNotFoundError(
        f"Config not found. Tried: {args_config}, configs/{checkpoint_name}.json"
    )


# ============================================================================
# Model Loading
# ============================================================================

def load_sft_model(checkpoint_dir, cfg, device="cuda"):
    """
    Load the full SFT model: Thinker + AudioEncoder + VisionEncoder + projectors.
    Returns (thinker, audio_enc, vision_enc, proj_a, proj_v, tokenizer, mel_spec).
    """
    thinker_cfg = cfg.get("thinker", {})
    ctx_len = cfg.get("ctx_len", 512)

    # --- Thinker ---
    thinker_ckpt_dir = cfg.get("thinker_ckpt", "checkpoints/thinker_tiny")
    tok_path = os.path.join(thinker_ckpt_dir, "tokenizer.model")
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")
    tokenizer = BPETokenizer(tok_path)
    vocab_size = thinker_cfg.get("vocab_size", tokenizer.sp.get_piece_size())

    thinker = ThinkerLM(
        vocab=vocab_size,
        n_layers=thinker_cfg.get("n_layers", 4),
        d=thinker_cfg.get("d_model", 256),
        heads=thinker_cfg.get("n_heads", 4),
        ff=thinker_cfg.get("d_ff", 1024),
        dropout=thinker_cfg.get("dropout", 0.1),
        rope_theta=thinker_cfg.get("rope_theta", 10000),
        ctx=ctx_len,
        use_gqa=thinker_cfg.get("use_gqa", False),
        use_swiglu=thinker_cfg.get("use_swiglu", True),
        use_moe=thinker_cfg.get("use_moe", False),
        num_experts=thinker_cfg.get("num_experts", 8),
        num_experts_per_tok=thinker_cfg.get("num_experts_per_tok", 2),
        compile_model=False,
        use_spiking=thinker_cfg.get("use_spiking", False),
        use_ltc=thinker_cfg.get("use_ltc", False),
        window_size=thinker_cfg.get("window_size", 0)
    ).to(device)

    # --- Audio Encoder ---
    audio_cfg_path = "configs/audio_enc_tiny.json"
    if os.path.exists(audio_cfg_path):
        audio_cfg = json.load(open(audio_cfg_path))
        downsample_factor = audio_cfg.get("downsample_time", 8)
        audio_enc = AudioEncoderTiny(
            d=audio_cfg.get("d_model", 192),
            heads=audio_cfg.get("n_heads", 3),
            ff=audio_cfg.get("d_ff", 768),
            layers=audio_cfg.get("n_layers", 4),
            dropout=audio_cfg.get("dropout", 0.1),
            downsample_factor=downsample_factor,
            use_spiking=audio_cfg.get("use_spiking", False),
            use_ltc=audio_cfg.get("use_ltc", False)
        ).to(device)
        audio_dim = audio_cfg.get("d_model", 192)
    else:
        audio_enc = AudioEncoderTiny(use_spiking=False, use_ltc=False).to(device)
        audio_dim = 384

    # --- Vision Encoder ---
    vision_cfg_path = "configs/vision_tiny.json"
    if os.path.exists(vision_cfg_path):
        vision_cfg = json.load(open(vision_cfg_path))
        vision_enc = ViTTiny(
            img_size=vision_cfg.get("img_size", 224),
            patch=vision_cfg.get("patch", 16),
            d=vision_cfg.get("d_model", 128),
            layers=vision_cfg.get("n_layers", 4),
            heads=vision_cfg.get("n_heads", 2),
            ff=vision_cfg.get("d_ff", 512),
            dropout=vision_cfg.get("dropout", 0.1)
        ).to(device)
        vision_dim = vision_cfg.get("d_model", 128)
    else:
        vision_enc = ViTTiny().to(device)
        vision_dim = 192

    # --- Projectors ---
    thinker_d_model = thinker_cfg.get("d_model", 256)
    proj_a = nn.Linear(audio_dim, thinker_d_model).to(device)
    proj_v = nn.Linear(vision_dim, thinker_d_model).to(device)

    # --- Load SFT checkpoint ---
    ckpt_path = os.path.join(checkpoint_dir, "omni.pt")
    if not os.path.exists(ckpt_path):
        # Try step checkpoints
        _, ckpt_data = find_checkpoint(checkpoint_dir, "omni.pt", "omni_step_", device)
        if ckpt_data is None:
            raise FileNotFoundError(f"No SFT checkpoint found in: {checkpoint_dir}")
        checkpoint = ckpt_data
    else:
        checkpoint = torch.load(ckpt_path, map_location=device)

    print(f"Loading SFT checkpoint from: {checkpoint_dir}")
    print(f"  Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'raw state_dict'}")

    if isinstance(checkpoint, dict):
        if "thinker" in checkpoint:
            thinker.load_state_dict(strip_orig_mod(checkpoint["thinker"]), strict=False)
            print("  Loaded thinker weights from SFT checkpoint")
        if "proj_a" in checkpoint:
            proj_a.load_state_dict(strip_orig_mod(checkpoint["proj_a"]), strict=False)
            print("  Loaded audio projector weights")
        if "proj_v" in checkpoint:
            proj_v.load_state_dict(strip_orig_mod(checkpoint["proj_v"]), strict=False)
            print("  Loaded vision projector weights")

    # Load pretrained encoder weights (the SFT checkpoint may not contain them)
    audio_ckpt_dir = cfg.get("audio_ckpt", "checkpoints/audio_enc_tiny")
    audio_enc_path = os.path.join(audio_ckpt_dir, "audio_enc.pt")
    if os.path.exists(audio_enc_path):
        audio_state = torch.load(audio_enc_path, map_location=device)
        audio_state = strip_orig_mod(audio_state.get("enc", audio_state))
        audio_enc.load_state_dict(audio_state, strict=False)
        print(f"  Loaded audio encoder from {audio_ckpt_dir}")

    vision_ckpt_dir = cfg.get("vision_ckpt", "checkpoints/vision_tiny")
    vision_enc_path = os.path.join(vision_ckpt_dir, "vision.pt")
    if os.path.exists(vision_enc_path):
        vision_state = torch.load(vision_enc_path, map_location=device)
        vision_state = strip_orig_mod(vision_state.get("vit", vision_state))
        vision_enc.load_state_dict(vision_state, strict=False)
        print(f"  Loaded vision encoder from {vision_ckpt_dir}")

    # Set all to eval
    thinker.eval()
    audio_enc.eval()
    vision_enc.eval()
    proj_a.eval()
    proj_v.eval()

    # Mel spectrogram transform
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=160, win_length=400, n_mels=128
    ).to(device)

    print(f"  Thinker d_model: {thinker_d_model}, ctx_len: {ctx_len}")
    print(f"  Audio encoder dim: {audio_dim}, Vision encoder dim: {vision_dim}")

    return thinker, audio_enc, vision_enc, proj_a, proj_v, tokenizer, mel_spec


# ============================================================================
# Data Loading Helpers
# ============================================================================

def load_text_samples(cfg, num_samples=50, seed=42):
    """Load random text samples from training data."""
    sft_mix = cfg.get("sft_mix", {})
    text_path = sft_mix.get("text_path", "data/text/production_corpus.txt")
    if not os.path.exists(text_path):
        print(f"  Warning: text file not found: {text_path}")
        return []
    with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [l.strip() for l in f if len(l.strip()) >= 10]
    rng = random.Random(seed)
    return rng.sample(lines, min(num_samples, len(lines)))


def load_image_samples(cfg, num_samples=20, seed=42):
    """Load random image samples from training manifest."""
    sft_mix = cfg.get("sft_mix", {})
    manifest_path = sft_mix.get("image_manifest", "data/images/production_annotations.json")
    image_root = sft_mix.get("image_root", "data/images")
    if not os.path.exists(manifest_path):
        print(f"  Warning: image manifest not found: {manifest_path}")
        return []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    # Filter to images that exist
    valid = []
    for item in manifest:
        img_path = os.path.join(image_root, item["image"])
        if os.path.exists(img_path):
            valid.append({"path": img_path, "caption": item.get("caption", "")})
    rng = random.Random(seed)
    return rng.sample(valid, min(num_samples, len(valid)))


def load_audio_samples(cfg, num_samples=20, seed=42):
    """Load random audio samples from ASR CSV."""
    sft_mix = cfg.get("sft_mix", {})
    asr_csv = sft_mix.get("asr_csv", "data/audio/production_asr.csv")
    if not os.path.exists(asr_csv):
        print(f"  Warning: ASR CSV not found: {asr_csv}")
        return []
    rows = []
    with open(asr_csv, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            wav_path = row.get("wav", "")
            if wav_path and os.path.exists(wav_path):
                rows.append({"path": wav_path, "text": row.get("text", "")})
    rng = random.Random(seed)
    return rng.sample(rows, min(num_samples, len(rows)))


# ============================================================================
# Evaluation Functions
# ============================================================================

def pack_text(prompt, answer, ctx_len, tokenizer):
    """Pack prompt+answer into token IDs with padding, return (x, y)."""
    ids = [1] + tokenizer.encode(prompt + " " + answer)
    ids = ids[:ctx_len]
    x = torch.tensor(ids + [0] * (ctx_len - len(ids)), dtype=torch.long)
    y = x.clone()
    y[:-1] = x[1:]
    y[-1] = 0
    return x, y


def evaluate_text_only(thinker, tokenizer, cfg, device, num_samples=50, verbose=True):
    """
    Test text-only inference: encode prompt, forward through thinker, compute loss.
    """
    prompt = cfg.get("prompt", "You are an omni assistant.")
    ctx_len = cfg.get("ctx_len", 512)
    loss_fn = CrossEntropyLoss(ignore_index=0)

    samples = load_text_samples(cfg, num_samples)
    if not samples:
        return None

    total_loss = 0.0
    total_tokens = 0
    correct_top1 = 0
    correct_top5 = 0
    total_predictions = 0

    iterator = tqdm(samples, desc="Text-only") if verbose else samples

    with torch.inference_mode():
        for text in iterator:
            try:
                x, y = pack_text(prompt, text, ctx_len, tokenizer)
                x, y = x.unsqueeze(0).to(device), y.to(device)

                logits = thinker(x)  # (1, T, vocab)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

                # Count non-padding tokens
                mask = y != 0
                n_tok = mask.sum().item()
                if n_tok == 0:
                    continue

                total_loss += loss.item() * n_tok
                total_tokens += n_tok

                # Top-1 accuracy
                preds = logits.argmax(dim=-1).squeeze(0)  # (T,)
                correct_top1 += ((preds == y) & mask).sum().item()

                # Top-5 accuracy
                top5 = logits.squeeze(0).topk(5, dim=-1).indices  # (T, 5)
                top5_correct = (top5 == y.unsqueeze(-1)).any(dim=-1) & mask
                correct_top5 += top5_correct.sum().item()

                total_predictions += n_tok

            except Exception as e:
                if verbose:
                    print(f"\n  Warning: {e}")
                continue

    if total_tokens == 0:
        return None

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss) if avg_loss < 20 else float('inf')

    return {
        'num_samples': len(samples),
        'total_tokens': total_tokens,
        'avg_loss': avg_loss,
        'perplexity': perplexity,
        'top1_accuracy': correct_top1 / total_predictions if total_predictions > 0 else 0.0,
        'top5_accuracy': correct_top5 / total_predictions if total_predictions > 0 else 0.0,
    }


def evaluate_image_text(thinker, vision_enc, proj_v, tokenizer, cfg, device, num_samples=20, verbose=True):
    """
    Test image+text inference: encode image through ViT + projector,
    prepend to text embeddings, forward through thinker, compute loss.
    """
    prompt = cfg.get("prompt", "You are an omni assistant.")
    ctx_len = cfg.get("ctx_len", 512)
    loss_fn = CrossEntropyLoss(ignore_index=0)
    img_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    samples = load_image_samples(cfg, num_samples)
    if not samples:
        return None

    total_loss = 0.0
    total_tokens = 0
    correct_top1 = 0
    correct_top5 = 0
    num_valid = 0

    iterator = tqdm(samples, desc="Image+Text") if verbose else samples

    with torch.inference_mode():
        for item in iterator:
            try:
                # Encode image
                img = img_transform(Image.open(item["path"]).convert("RGB")).unsqueeze(0).to(device)
                cls_emb, _ = vision_enc(img)        # (1, 1, d_vision)
                img_emb = proj_v(cls_emb)            # (1, 1, d_thinker)

                # Encode text
                answer = item["caption"] if item["caption"] else "An image."
                x_ids, y_ids = pack_text(prompt, answer, ctx_len, tokenizer)
                x_ids = x_ids.to(device)
                y_ids = y_ids.to(device)

                text_emb = thinker.tok_emb(x_ids.unsqueeze(0))  # (1, T_text, d_thinker)

                # Combine: [image_emb | text_emb] truncated to ctx_len
                img_len = img_emb.shape[1]
                max_text = ctx_len - img_len
                if max_text < 1:
                    max_text = 1
                text_emb = text_emb[:, :max_text, :]
                y_ids = y_ids[:max_text]

                combined_emb = torch.cat([img_emb, text_emb], dim=1)  # (1, T_total, d)

                # Build targets: zeros for image tokens, then text targets
                img_padding = torch.zeros(img_len, dtype=y_ids.dtype, device=device)
                combined_y = torch.cat([img_padding, y_ids], dim=0)  # (T_total,)

                # Build causal attention mask
                T = combined_emb.shape[1]
                attn_mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0)  # (1, T, T)

                logits = thinker(embeddings=combined_emb, attn_mask=attn_mask)
                loss = loss_fn(logits.view(-1, logits.size(-1)), combined_y.view(-1))

                mask = combined_y != 0
                n_tok = mask.sum().item()
                if n_tok == 0:
                    continue

                total_loss += loss.item() * n_tok
                total_tokens += n_tok
                num_valid += 1

                # Top-1 accuracy
                preds = logits.squeeze(0).argmax(dim=-1)  # (T_total,)
                correct_top1 += ((preds == combined_y) & mask).sum().item()

                # Top-5 accuracy
                top5 = logits.squeeze(0).topk(5, dim=-1).indices  # (T_total, 5)
                top5_correct = (top5 == combined_y.unsqueeze(-1)).any(dim=-1) & mask
                correct_top5 += top5_correct.sum().item()

            except Exception as e:
                if verbose:
                    print(f"\n  Warning (image): {e}")
                continue

    if total_tokens == 0:
        return None

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss) if avg_loss < 20 else float('inf')

    return {
        'num_samples': num_valid,
        'total_tokens': total_tokens,
        'avg_loss': avg_loss,
        'perplexity': perplexity,
        'top1_accuracy': correct_top1 / total_tokens if total_tokens > 0 else 0.0,
        'top5_accuracy': correct_top5 / total_tokens if total_tokens > 0 else 0.0,
    }


def evaluate_audio_text(thinker, audio_enc, proj_a, tokenizer, mel_spec, cfg, device, num_samples=20, verbose=True):
    """
    Test audio+text inference: encode audio through AudioEncoder + projector,
    prepend to text embeddings, forward through thinker, compute loss.
    """
    prompt = cfg.get("prompt", "You are an omni assistant.")
    ctx_len = cfg.get("ctx_len", 512)
    loss_fn = CrossEntropyLoss(ignore_index=0)

    samples = load_audio_samples(cfg, num_samples)
    if not samples:
        return None

    total_loss = 0.0
    total_tokens = 0
    correct_top1 = 0
    correct_top5 = 0
    num_valid = 0

    iterator = tqdm(samples, desc="Audio+Text") if verbose else samples

    with torch.inference_mode():
        for item in iterator:
            try:
                # Encode audio -> mel -> encoder -> projector
                wav, _ = load_audio(item["path"])
                wav = wav.to(device)
                mel = mel_spec(wav)[0].T.unsqueeze(0)  # (1, T_mel, 128)
                audio_emb = audio_enc(mel)              # (1, T', d_audio)
                audio_emb = proj_a(audio_emb)           # (1, T', d_thinker)

                # Limit audio tokens
                max_audio_tokens = ctx_len // 4
                audio_emb = audio_emb[:, :max_audio_tokens, :]
                audio_len = audio_emb.shape[1]

                # Encode text
                answer = item["text"] if item["text"] else "Audio content."
                x_ids, y_ids = pack_text(prompt, answer, ctx_len, tokenizer)
                x_ids = x_ids.to(device)
                y_ids = y_ids.to(device)

                text_emb = thinker.tok_emb(x_ids.unsqueeze(0))  # (1, T_text, d_thinker)

                # Combine: [audio_emb | text_emb] truncated to ctx_len
                max_text = ctx_len - audio_len
                if max_text < 1:
                    max_text = 1
                text_emb = text_emb[:, :max_text, :]
                y_ids = y_ids[:max_text]

                combined_emb = torch.cat([audio_emb, text_emb], dim=1)

                # Build targets
                audio_padding = torch.zeros(audio_len, dtype=y_ids.dtype, device=device)
                combined_y = torch.cat([audio_padding, y_ids], dim=0)

                # Build causal attention mask
                T = combined_emb.shape[1]
                attn_mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0)

                logits = thinker(embeddings=combined_emb, attn_mask=attn_mask)
                loss = loss_fn(logits.view(-1, logits.size(-1)), combined_y.view(-1))

                mask = combined_y != 0
                n_tok = mask.sum().item()
                if n_tok == 0:
                    continue

                total_loss += loss.item() * n_tok
                total_tokens += n_tok
                num_valid += 1

                # Top-1 accuracy
                preds = logits.squeeze(0).argmax(dim=-1)  # (T_total,)
                correct_top1 += ((preds == combined_y) & mask).sum().item()

                # Top-5 accuracy
                top5 = logits.squeeze(0).topk(5, dim=-1).indices  # (T_total, 5)
                top5_correct = (top5 == combined_y.unsqueeze(-1)).any(dim=-1) & mask
                correct_top5 += top5_correct.sum().item()

            except Exception as e:
                if verbose:
                    print(f"\n  Warning (audio): {e}")
                continue

    if total_tokens == 0:
        return None

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss) if avg_loss < 20 else float('inf')

    return {
        'num_samples': num_valid,
        'total_tokens': total_tokens,
        'avg_loss': avg_loss,
        'perplexity': perplexity,
        'top1_accuracy': correct_top1 / total_tokens if total_tokens > 0 else 0.0,
        'top5_accuracy': correct_top5 / total_tokens if total_tokens > 0 else 0.0,
    }


# ============================================================================
# Results Reporting
# ============================================================================

def print_results(text_metrics, image_metrics, audio_metrics):
    """Pretty print evaluation results for all modalities."""
    print(f"\n{'='*70}")
    print("SFT (MULTIMODAL) EVALUATION RESULTS")
    print(f"{'='*70}")

    modalities_working = []

    # --- Text-only ---
    print(f"\nTEXT-ONLY INFERENCE:")
    if text_metrics:
        print(f"  Samples: {text_metrics['num_samples']}")
        print(f"  Tokens evaluated: {text_metrics['total_tokens']:,}")
        print(f"  Average Loss: {text_metrics['avg_loss']:.4f}")
        print(f"  Perplexity: {text_metrics['perplexity']:.2f}")
        print(f"  Top-1 Accuracy: {text_metrics['top1_accuracy']*100:.2f}%")
        print(f"  Top-5 Accuracy: {text_metrics['top5_accuracy']*100:.2f}%")
        modalities_working.append("text")
    else:
        print(f"  SKIPPED - No text data available")

    # --- Image+Text ---
    print(f"\nIMAGE+TEXT INFERENCE:")
    if image_metrics:
        print(f"  Samples: {image_metrics['num_samples']}")
        print(f"  Tokens evaluated: {image_metrics['total_tokens']:,}")
        print(f"  Average Loss: {image_metrics['avg_loss']:.4f}")
        print(f"  Perplexity: {image_metrics['perplexity']:.2f}")
        print(f"  Top-1 Accuracy: {image_metrics['top1_accuracy']*100:.2f}%")
        print(f"  Top-5 Accuracy: {image_metrics['top5_accuracy']*100:.2f}%")
        modalities_working.append("image+text")
    else:
        print(f"  SKIPPED - No image data available")

    # --- Audio+Text ---
    print(f"\nAUDIO+TEXT INFERENCE:")
    if audio_metrics:
        print(f"  Samples: {audio_metrics['num_samples']}")
        print(f"  Tokens evaluated: {audio_metrics['total_tokens']:,}")
        print(f"  Average Loss: {audio_metrics['avg_loss']:.4f}")
        print(f"  Perplexity: {audio_metrics['perplexity']:.2f}")
        print(f"  Top-1 Accuracy: {audio_metrics['top1_accuracy']*100:.2f}%")
        print(f"  Top-5 Accuracy: {audio_metrics['top5_accuracy']*100:.2f}%")
        modalities_working.append("audio+text")
    else:
        print(f"  SKIPPED - No audio data available")

    # --- Interpretation ---
    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print(f"{'='*70}")

    if text_metrics:
        ppl = text_metrics['perplexity']
        if ppl < 15:
            print(f"  Text perplexity: EXCELLENT ({ppl:.2f})")
        elif ppl < 50:
            print(f"  Text perplexity: GOOD ({ppl:.2f})")
        elif ppl < 150:
            print(f"  Text perplexity: ACCEPTABLE ({ppl:.2f})")
        else:
            print(f"  Text perplexity: POOR ({ppl:.2f}) - needs more training")

    if image_metrics:
        ppl = image_metrics['perplexity']
        if ppl < 20:
            print(f"  Image+text perplexity: EXCELLENT ({ppl:.2f})")
        elif ppl < 80:
            print(f"  Image+text perplexity: GOOD ({ppl:.2f})")
        elif ppl < 200:
            print(f"  Image+text perplexity: ACCEPTABLE ({ppl:.2f})")
        else:
            print(f"  Image+text perplexity: POOR ({ppl:.2f}) - needs more training")

    if audio_metrics:
        ppl = audio_metrics['perplexity']
        if ppl < 20:
            print(f"  Audio+text perplexity: EXCELLENT ({ppl:.2f})")
        elif ppl < 80:
            print(f"  Audio+text perplexity: GOOD ({ppl:.2f})")
        elif ppl < 200:
            print(f"  Audio+text perplexity: ACCEPTABLE ({ppl:.2f})")
        else:
            print(f"  Audio+text perplexity: POOR ({ppl:.2f}) - needs more training")

    # Multimodal features check
    print(f"\nMULTIMODAL FEATURES:")
    print(f"  Modalities tested: {', '.join(modalities_working) if modalities_working else 'NONE'}")
    if len(modalities_working) == 3:
        print(f"  All 3 modalities operational!")
    elif len(modalities_working) >= 1:
        print(f"  {len(modalities_working)}/3 modalities operational")
        missing = set(["text", "image+text", "audio+text"]) - set(modalities_working)
        print(f"  Missing: {', '.join(missing)}")
    else:
        print(f"  No modalities could be tested")

    # Overall assessment
    print(f"\n{'='*70}")
    all_good = True
    if text_metrics and text_metrics['perplexity'] < 50:
        pass
    else:
        all_good = False
    if image_metrics and image_metrics['perplexity'] < 200:
        pass
    else:
        all_good = False
    if audio_metrics and audio_metrics['perplexity'] < 200:
        pass
    else:
        all_good = False

    if all_good:
        print("SFT model is working well across all modalities!")
        print("Ready for multimodal inference.")
    elif len(modalities_working) >= 2:
        print("SFT model is partially working.")
        print("Some modalities may need more training or data.")
    else:
        print("SFT model needs more training.")
        print("Check data paths and hyperparameters.")
    print(f"{'='*70}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test SFT (multimodal) model accuracy")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/omni_sft_tiny",
                        help="Path to SFT checkpoint directory")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config JSON (fallback: configs/<checkpoint_name>.json)")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of text samples to evaluate (default: 50)")
    parser.add_argument("--num_mm_samples", type=int, default=20,
                        help="Number of multimodal samples per modality (default: 20)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with fewer samples")
    parser.add_argument("--log_file", default=default_log_path(__file__), help="Write stdout/stderr to this file (UTF-8)")
    args = parser.parse_args()
    enable_log_file(args.log_file, header=f"test_sft.py start | checkpoint={args.checkpoint}")

    if args.quick:
        args.num_samples = 10
        args.num_mm_samples = 5

    print("=" * 70)
    print("SFT (MULTIMODAL) ACCURACY TEST")
    print("=" * 70)

    if args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Resolve config
    try:
        cfg = resolve_config(args.config, args.checkpoint)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Load model
    try:
        thinker, audio_enc, vision_enc, proj_a, proj_v, tokenizer, mel_spec = \
            load_sft_model(args.checkpoint, cfg, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Evaluate all modalities
    try:
        print(f"\n--- Evaluating text-only inference ---")
        text_metrics = evaluate_text_only(
            thinker, tokenizer, cfg, args.device,
            num_samples=args.num_samples, verbose=True
        )

        print(f"\n--- Evaluating image+text inference ---")
        image_metrics = evaluate_image_text(
            thinker, vision_enc, proj_v, tokenizer, cfg, args.device,
            num_samples=args.num_mm_samples, verbose=True
        )

        print(f"\n--- Evaluating audio+text inference ---")
        audio_metrics = evaluate_audio_text(
            thinker, audio_enc, proj_a, tokenizer, mel_spec, cfg, args.device,
            num_samples=args.num_mm_samples, verbose=True
        )

        print_results(text_metrics, image_metrics, audio_metrics)

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
