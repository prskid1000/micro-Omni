"""
Unified post-training script for continuing training from a checkpoint with a different dataset.

Supports all model types:
- Thinker (text LLM)
- Audio Encoder (ASR)
- Vision Encoder (image-caption)
- Talker (TTS)

Before using this script, prepare your data with:
    python scripts/prep_post_training_data.py --input <your_data> --output <output_path> --format <text|audio_asr|audio_tts|images>

Usage:
    # Thinker (text)
    python post_train.py \
        --config configs/thinker_tiny.json \
        --checkpoint checkpoints/thinker_tiny/thinker.pt \
        --new_dataset data/post_training/text.txt

    # Audio Encoder (ASR)
    python post_train.py \
        --config configs/audio_enc_tiny.json \
        --checkpoint checkpoints/audio_enc_tiny/audio_enc.pt \
        --new_dataset data/post_training/asr.csv

    # Vision Encoder
    python post_train.py \
        --config configs/vision_tiny.json \
        --checkpoint checkpoints/vision_tiny/vision.pt \
        --new_dataset data/post_training/images.json

    # Talker (TTS)
    python post_train.py \
        --config configs/talker_tiny.json \
        --checkpoint checkpoints/talker_tiny/talker.pt \
        --new_dataset data/post_training/tts.csv

For detailed documentation, see: study/14_Post_Training.md
"""

import argparse
import json
import os
import csv
import torch
import gc
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchaudio
import json as js

# Model imports
from omni.thinker import ThinkerLM
from omni.audio_encoder import AudioEncoderTiny
from omni.vision_encoder import ViTTiny
from omni.talker import TalkerTiny
from omni.codec import RVQ
from omni.tokenizer import BPETokenizer
from omni.training_utils import set_seed, get_lr_scheduler, validate_loss, check_gradient_explosion, SimpleLogger, reload_from_last_checkpoint, cleanup_old_checkpoints
from tqdm import tqdm

# ============================================================================
# Dataset Classes
# ============================================================================

class TextDataset(Dataset):
    def __init__(self, path, tokenizer, ctx):
        with open(path, 'r', encoding='utf-8') as f:
            self.lines = [l.strip() for l in f if l.strip()]
        self.tok = tokenizer
        self.ctx = ctx
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, i):
        ids = self.tok.encode(self.lines[i])[:self.ctx-1]
        ids = [1] + ids  # BOS=1
        pad = [0] * (self.ctx - len(ids))
        x = torch.tensor(ids + pad, dtype=torch.long)
        y = x.clone()
        y[:-1] = x[1:]
        y[-1] = 0
        return x, y

class ASRDataset(Dataset):
    def __init__(self, csv_path, sr=16000, n_mels=128, cfg=None):
        self.rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            rd = csv.DictReader(f)
            for r in rd:
                self.rows.append(r)
        self.sr = sr
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, hop_length=160, win_length=400, n_mels=n_mels
        )
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, i):
        path, text = self.rows[i]["wav"], self.rows[i]["text"]
        wav, sr = torchaudio.load(path)
        assert sr == self.sr
        mel = self.melspec(wav)[0].T  # (T, n_mels)
        return mel, text

def collate_fn_asr(batch):
    """Collate function for ASR (variable-length mel spectrograms)"""
    mels, texts = zip(*batch)
    max_len = max(m.shape[0] for m in mels)
    n_mels = mels[0].shape[1]
    padded_mels = []
    for m in mels:
        pad_len = max_len - m.shape[0]
        if pad_len > 0:
            m = torch.cat([m, torch.zeros(pad_len, n_mels)], dim=0)
        padded_mels.append(m)
    return torch.stack(padded_mels), list(texts)

class ImgCapDataset(Dataset):
    def __init__(self, manifest, image_root, img_size=224):
        self.items = js.load(open(manifest, 'r', encoding='utf-8'))
        self.root = image_root
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, i):
        it = self.items[i]
        img_path = it["image"] if os.path.isabs(it["image"]) else os.path.join(self.root, it["image"])
        img = Image.open(img_path).convert("RGB")
        return self.tf(img), it["caption"]

class TTSDataset(Dataset):
    def __init__(self, csv_path, sr=16000, n_mels=128, frame_ms=80, cfg=None):
        self.rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            rd = csv.DictReader(f)
            for r in rd:
                self.rows.append(r)
        self.sr = sr
        hop_length = int(sr * frame_ms / 1000)
        win_length = min(1024, hop_length * 4)
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, hop_length=hop_length, 
            win_length=win_length, n_mels=n_mels
        )
        self.frame = int(sr * 0.08)
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, i):
        text, path = self.rows[i]["text"], self.rows[i]["wav"]
        wav, sr = torchaudio.load(path)
        assert sr == self.sr
        mel = self.melspec(wav)[0].T  # (T, n_mels)
        return mel

def collate_fn_tts(batch):
    """Collate function for TTS (variable-length mel spectrograms)"""
    max_len = max(m.shape[0] for m in batch)
    n_mels = batch[0].shape[1]
    padded = []
    for m in batch:
        pad_len = max_len - m.shape[0]
        if pad_len > 0:
            m = torch.cat([m, torch.zeros(pad_len, n_mels)], dim=0)
        padded.append(m)
    return torch.stack(padded)

# ============================================================================
# Model Type Detection
# ============================================================================

def detect_model_type(cfg):
    """Detect model type from config"""
    if "vocab_size" in cfg and "ctx_len" in cfg:
        return "thinker"
    elif "train_csv" in cfg or "ctc_vocab_size" in cfg:
        return "audio_enc"
    elif "train_manifest" in cfg or "image_root" in cfg:
        return "vision"
    elif "tts_csv" in cfg or "codebooks" in cfg:
        return "talker"
    else:
        raise ValueError("Could not detect model type from config. Please specify --model_type")

def load_checkpoint(checkpoint_path, device, model_type):
    """Load checkpoint and return model state dict(s) and metadata"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        # Try to find model state dict based on model type
        model_keys = {
            "thinker": ["model", "thinker"],
            "audio_enc": ["enc", "model"],
            "vision": ["vit", "model"],
            "talker": ["talker", "model"]
        }
        
        model_state = None
        for key in model_keys.get(model_type, ["model"]):
            if key in checkpoint:
                model_state = checkpoint[key]
                break
        
        if model_state is None:
            # Assume it's a model state dict
            model_state = checkpoint
        
        # For talker, also load RVQ if present
        rvq_state = None
        if model_type == "talker" and "rvq" in checkpoint:
            rvq_state = checkpoint["rvq"]
        
        # For audio_enc, also load head if present
        head_state = None
        if model_type == "audio_enc" and "head" in checkpoint:
            head_state = checkpoint["head"]
        
        # For vision, also load projectors if present
        img_proj_state = None
        text_proj_state = None
        text_embed_state = None
        if model_type == "vision":
            if "img_proj" in checkpoint:
                img_proj_state = checkpoint["img_proj"]
            if "text_proj" in checkpoint:
                text_proj_state = checkpoint["text_proj"]
            if "text_embed" in checkpoint:
                text_embed_state = checkpoint["text_embed"]
        
        return model_state, checkpoint, {
            "rvq": rvq_state,
            "head": head_state,
            "img_proj": img_proj_state,
            "text_proj": text_proj_state,
            "text_embed": text_embed_state
        }
    else:
        # Legacy format - just model weights
        return checkpoint, {}, {}

# ============================================================================
# Training Functions (modular)
# ============================================================================

def train_thinker(cfg, checkpoint_path, new_dataset, args, device):
    """Post-train Thinker model"""
    # Load tokenizer (check both original and post-training directories)
    original_save_dir = os.path.dirname(checkpoint_path)
    spm_model_original = os.path.join(original_save_dir, "tokenizer.model")
    spm_model_post = os.path.join(cfg["save_dir"], "tokenizer.model")
    
    if os.path.exists(spm_model_original):
        spm_model = spm_model_original
        print(f"Using existing tokenizer from: {spm_model}")
    elif os.path.exists(spm_model_post):
        spm_model = spm_model_post
        print(f"Using existing tokenizer from: {spm_model}")
    else:
        # Train new tokenizer in post-training directory
        spm_model = spm_model_post
        print(f"Training new tokenizer from {new_dataset}...")
        BPETokenizer.train_new(new_dataset, spm_model, vocab_size=cfg["vocab_size"])
    tok = BPETokenizer(spm_model)
    
    # Load dataset
    ds = TextDataset(new_dataset, tok, cfg["ctx_len"])
    val_split = cfg.get("val_split", 0.1)
    total_size = len(ds)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.get("seed", 42))
    )
    train_dl = DataLoader(train_ds, batch_size=cfg.get("batch_size", 8), shuffle=True,
                         num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True))
    val_dl = DataLoader(val_ds, batch_size=cfg.get("batch_size", 8), shuffle=False,
                       num_workers=cfg.get("num_workers", 2), drop_last=False)
    
    # Create model
    # torch.compile() support (optional, PyTorch 2.0+)
    use_compile = cfg.get("use_compile", False)
    
    model = ThinkerLM(
        cfg["vocab_size"], cfg["n_layers"], cfg["d_model"], cfg["n_heads"],
        cfg["d_ff"], cfg["dropout"], cfg["rope_theta"], cfg["ctx_len"],
        use_gqa=cfg.get("use_gqa", False), use_swiglu=cfg.get("use_swiglu", True),
        use_moe=cfg.get("use_moe", False), num_experts=cfg.get("num_experts", 8),
        num_experts_per_tok=cfg.get("num_experts_per_tok", 2),
        compile_model=use_compile
    ).to(device)
    
    # Load checkpoint
    model_state, checkpoint_meta, _ = load_checkpoint(checkpoint_path, device, "thinker")
    try:
        model.load_state_dict(model_state, strict=True)
        print("✓ Model weights loaded successfully")
    except RuntimeError as e:
        print(f"⚠ Warning: {e}")
        model.load_state_dict(model_state, strict=False)
        print("✓ Model weights loaded (some keys may be missing)")
    
    # Setup optimizer and loss
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    
    return model, opt, loss_fn, train_dl, val_dl, checkpoint_meta, "thinker", None, None, None, None

def train_audio_enc(cfg, checkpoint_path, new_dataset, args, device):
    """Post-train Audio Encoder model"""
    # Load dataset
    sr = cfg.get("sample_rate", 16000)
    n_mels = cfg.get("mel_bins", 128)
    ds = ASRDataset(new_dataset, sr=sr, n_mels=n_mels, cfg=cfg)
    val_split = cfg.get("val_split", 0.1)
    total_size = len(ds)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.get("seed", 42))
    )
    train_dl = DataLoader(train_ds, batch_size=cfg.get("batch_size", 4), shuffle=True,
                          num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True),
                          collate_fn=collate_fn_asr)
    val_dl = DataLoader(val_ds, batch_size=cfg.get("batch_size", 4), shuffle=False,
                       num_workers=cfg.get("num_workers", 2), drop_last=False,
                       collate_fn=collate_fn_asr)
    
    # Create model
    downsample_factor = cfg.get("downsample_time", 8)
    # torch.compile() support (optional, PyTorch 2.0+)
    use_compile = cfg.get("use_compile", False)
    
    model = AudioEncoderTiny(
        cfg["d_model"], cfg["n_heads"], cfg["d_ff"], cfg["n_layers"],
        cfg["dropout"], downsample_factor=downsample_factor,
        compile_model=use_compile
    ).to(device)
    
    # Build character vocabulary for CTC
    char_to_idx = {}
    for i in range(32, 127):  # Printable ASCII
        char_to_idx[chr(i)] = len(char_to_idx) + 1
    char_to_idx['\n'] = len(char_to_idx) + 1
    char_to_idx['\t'] = len(char_to_idx) + 1
    char_to_idx['<UNK>'] = len(char_to_idx) + 1
    vocab_size_ctc = len(char_to_idx) + 1  # +1 for blank token (0)
    vocab = cfg.get("ctc_vocab_size", vocab_size_ctc)
    
    head = nn.Linear(cfg["d_model"], vocab).to(device)
    
    # Load checkpoint
    model_state, checkpoint_meta, extra_states = load_checkpoint(checkpoint_path, device, "audio_enc")
    try:
        model.load_state_dict(model_state, strict=True)
        print("✓ Model weights loaded successfully")
    except RuntimeError as e:
        print(f"⚠ Warning: {e}")
        model.load_state_dict(model_state, strict=False)
        print("✓ Model weights loaded (some keys may be missing)")
    
    if extra_states["head"] is not None:
        try:
            head.load_state_dict(extra_states["head"], strict=True)
            print("✓ Head weights loaded successfully")
        except RuntimeError as e:
            print(f"⚠ Warning: Could not load head weights: {e}")
    
    # Setup optimizer and loss
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                            lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    head.char_to_idx = char_to_idx
    head.max_text_len = cfg.get("max_text_len", 64)
    head.unk_idx = char_to_idx['<UNK>']
    
    return model, opt, loss_fn, train_dl, val_dl, checkpoint_meta, "audio_enc", head, None, None, None

def train_vision(cfg, checkpoint_path, new_dataset, args, device):
    """Post-train Vision Encoder model"""
    # Determine image root
    image_root = cfg.get("image_root", os.path.dirname(new_dataset))
    if not os.path.isabs(image_root):
        image_root = os.path.join(os.path.dirname(cfg.get("image_root", ".")), image_root)
    
    # Load dataset
    ds = ImgCapDataset(new_dataset, image_root, cfg["img_size"])
    val_split = cfg.get("val_split", 0.1)
    total_size = len(ds)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.get("seed", 42))
    )
    train_dl = DataLoader(train_ds, batch_size=cfg.get("batch_size", 8), shuffle=True,
                          num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True))
    val_dl = DataLoader(val_ds, batch_size=cfg.get("batch_size", 8), shuffle=False,
                       num_workers=cfg.get("num_workers", 2), drop_last=False)
    
    # Create model
    # torch.compile() support (optional, PyTorch 2.0+)
    use_compile = cfg.get("use_compile", False)
    
    model = ViTTiny(
        cfg["img_size"], cfg["patch"], cfg["d_model"], cfg["n_layers"],
        cfg["n_heads"], cfg["d_ff"], cfg["dropout"],
        compile_model=use_compile
    ).to(device)
    
    # Projectors for contrastive learning
    embed_dim = cfg.get("embed_dim", cfg["d_model"])
    img_proj = nn.Linear(cfg["d_model"], embed_dim).to(device)
    text_proj = nn.Linear(cfg["d_model"], embed_dim).to(device)
    vocab_size = cfg.get("vocab_size", 10000)
    text_embed = nn.Embedding(vocab_size, cfg["d_model"]).to(device)
    
    # Build vocabulary from captions
    all_captions = [item["caption"] for item in ds.items]
    word_to_idx = {}
    word_freq = {}
    for cap in all_captions:
        words = cap.lower().split()
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    vocab_size = min(vocab_size, len(sorted_words))
    word_to_idx = {word: idx+1 for idx, (word, _) in enumerate(sorted_words[:vocab_size-1])}
    word_to_idx["<UNK>"] = vocab_size - 1
    
    def encode_caption(caption):
        words = caption.lower().split()
        word_ids = [word_to_idx.get(word, word_to_idx["<UNK>"]) for word in words]
        if len(word_ids) == 0:
            word_ids = [word_to_idx["<UNK>"]]
        word_embeds = text_embed(torch.tensor(word_ids, device=device))
        return word_embeds.mean(dim=0)
    
    # Load checkpoint
    model_state, checkpoint_meta, extra_states = load_checkpoint(checkpoint_path, device, "vision")
    try:
        model.load_state_dict(model_state, strict=True)
        print("✓ Model weights loaded successfully")
    except RuntimeError as e:
        print(f"⚠ Warning: {e}")
        model.load_state_dict(model_state, strict=False)
        print("✓ Model weights loaded (some keys may be missing)")
    
    if extra_states["img_proj"] is not None:
        try:
            img_proj.load_state_dict(extra_states["img_proj"], strict=True)
            print("✓ Image projector loaded")
        except Exception as e:
            print(f"⚠ Warning: Could not load image projector: {e}")
    if extra_states["text_proj"] is not None:
        try:
            text_proj.load_state_dict(extra_states["text_proj"], strict=True)
            print("✓ Text projector loaded")
        except Exception as e:
            print(f"⚠ Warning: Could not load text projector: {e}")
    if extra_states["text_embed"] is not None:
        try:
            text_embed.load_state_dict(extra_states["text_embed"], strict=True)
            print("✓ Text embeddings loaded")
        except Exception as e:
            print(f"⚠ Warning: Could not load text embeddings: {e}")
    
    # Setup optimizer
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(img_proj.parameters()) +
        list(text_proj.parameters()) + list(text_embed.parameters()),
        lr=cfg["lr"], weight_decay=cfg["wd"]
    )
    temperature = cfg.get("temperature", 0.07)
    
    # Contrastive loss function
    def contrastive_loss(img_features, text_features, temperature):
        # Normalize features
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Compute similarity
        logits = torch.matmul(img_features, text_features.T) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = nn.CrossEntropyLoss()(logits, labels) + nn.CrossEntropyLoss()(logits.T, labels)
        return loss / 2
    
    loss_fn = lambda img_feat, txt_feat: contrastive_loss(img_feat, txt_feat, temperature)
    
    return model, opt, loss_fn, train_dl, val_dl, checkpoint_meta, "vision", None, img_proj, text_proj, (text_embed, encode_caption)

def train_talker(cfg, checkpoint_path, new_dataset, args, device):
    """Post-train Talker model"""
    # Load dataset
    sr = cfg.get("sample_rate", 16000)
    n_mels = cfg.get("n_mels", 128)
    frame_ms = cfg.get("frame_ms", 80)
    ds = TTSDataset(new_dataset, sr=sr, n_mels=n_mels, frame_ms=frame_ms, cfg=cfg)
    val_split = cfg.get("val_split", 0.1)
    total_size = len(ds)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.get("seed", 42))
    )
    train_dl = DataLoader(train_ds, batch_size=cfg.get("batch_size", 4), shuffle=True,
                          num_workers=cfg.get("num_workers", 2), drop_last=cfg.get("drop_last", True),
                          collate_fn=collate_fn_tts)
    val_dl = DataLoader(val_ds, batch_size=cfg.get("batch_size", 4), shuffle=False,
                       num_workers=cfg.get("num_workers", 2), drop_last=False,
                       collate_fn=collate_fn_tts)
    
    # Create models
    # torch.compile() support (optional, PyTorch 2.0+)
    use_compile = cfg.get("use_compile", False)
    
    rvq = RVQ(cfg["codebooks"], cfg["codebook_size"], d=64, compile_model=use_compile).to(device)
    
    model = TalkerTiny(
        cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["d_ff"],
        cfg["codebooks"], cfg["codebook_size"], cfg["dropout"],
        use_gqa=cfg.get("use_gqa", False), use_swiglu=cfg.get("use_swiglu", True),
        rope_theta=cfg.get("rope_theta", 10000.0),
        compile_model=use_compile
    ).to(device)
    
    # Load checkpoint
    model_state, checkpoint_meta, extra_states = load_checkpoint(checkpoint_path, device, "talker")
    try:
        model.load_state_dict(model_state, strict=True)
        print("✓ Talker weights loaded successfully")
    except RuntimeError as e:
        print(f"⚠ Warning: {e}")
        model.load_state_dict(model_state, strict=False)
        print("✓ Talker weights loaded (some keys may be missing)")
    
    if extra_states["rvq"] is not None:
        try:
            rvq.load_state_dict(extra_states["rvq"], strict=True)
            print("✓ RVQ weights loaded successfully")
        except RuntimeError as e:
            print(f"⚠ Warning: Could not load RVQ weights: {e}")
    
    # Setup optimizer and loss
    opt = torch.optim.AdamW(list(rvq.parameters()) + list(model.parameters()),
                            lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_fn = nn.CrossEntropyLoss()
    
    return model, opt, loss_fn, train_dl, val_dl, checkpoint_meta, "talker", rvq, None, None, None

# ============================================================================
# Main Training Loop (unified)
# ============================================================================

def run_training_loop(model, opt, loss_fn, train_dl, val_dl, checkpoint_meta, model_type,
                     head, rvq, img_proj, text_proj, text_embed_encode, cfg, args, device):
    """Unified training loop for all model types"""
    
    # Setup scheduler
    warmup_steps = cfg.get("warmup_steps", 500)
    max_steps = cfg.get("max_steps", 5000)
    scheduler = get_lr_scheduler(opt, warmup_steps, max_steps)
    
    # Initialize logger early for resumption detection
    logger = SimpleLogger("PostTrain")
    
    # Check for existing post-training checkpoints for automatic resumption
    checkpoint_prefix = f"{model_type}_post_step_"
    existing_checkpoints = []
    if os.path.exists(cfg["save_dir"]):
        for fname in os.listdir(cfg["save_dir"]):
            if fname.startswith(checkpoint_prefix) and fname.endswith(".pt"):
                try:
                    step_num = int(fname.replace(checkpoint_prefix, "").replace(".pt", ""))
                    existing_checkpoints.append((step_num, fname))
                except ValueError:
                    continue
    
    # If post-training checkpoints exist, resume from the latest one
    if existing_checkpoints and not args.reset_step:
        existing_checkpoints.sort(reverse=True)
        latest_step, latest_fname = existing_checkpoints[0]
        latest_checkpoint_path = os.path.join(cfg["save_dir"], latest_fname)
        
        logger.info(f"🔄 Found existing post-training checkpoint: {latest_fname}")
        logger.info(f"Resuming from step {latest_step}...")
        
        try:
            resume_checkpoint = torch.load(latest_checkpoint_path, map_location=device)
            
            # Load model state
            model_key = {"thinker": "model", "audio_enc": "enc", "vision": "vit", "talker": "talker"}[model_type]
            if model_key in resume_checkpoint:
                model.load_state_dict(resume_checkpoint[model_key])
                logger.info("✓ Model state resumed")
            
            # Load additional components
            if model_type == "audio_enc" and head and "head" in resume_checkpoint:
                head.load_state_dict(resume_checkpoint["head"])
                logger.info("✓ Head state resumed")
            elif model_type == "vision" and img_proj and text_proj:
                if "img_proj" in resume_checkpoint:
                    img_proj.load_state_dict(resume_checkpoint["img_proj"])
                if "text_proj" in resume_checkpoint:
                    text_proj.load_state_dict(resume_checkpoint["text_proj"])
                if "text_embed" in resume_checkpoint and text_embed_encode:
                    text_embed_encode[0].load_state_dict(resume_checkpoint["text_embed"])
                logger.info("✓ Projectors state resumed")
            elif model_type == "talker" and rvq and "rvq" in resume_checkpoint:
                rvq.load_state_dict(resume_checkpoint["rvq"])
                logger.info("✓ RVQ state resumed")
            
            # Load training state
            if not args.reset_optimizer and "optimizer" in resume_checkpoint:
                opt.load_state_dict(resume_checkpoint["optimizer"])
                logger.info("✓ Optimizer state resumed")
            
            if not args.reset_scheduler and "scheduler" in resume_checkpoint:
                scheduler.load_state_dict(resume_checkpoint["scheduler"])
                logger.info("✓ Scheduler state resumed")
            
            step = resume_checkpoint.get("step", latest_step)
            best_val_loss = resume_checkpoint.get("best_val_loss", float('inf'))
            
            logger.info(f"✓ Successfully resumed from step {step}")
            logger.info(f"✓ Best validation loss so far: {best_val_loss:.4f}")
            
            # Override checkpoint_meta with resumed checkpoint
            checkpoint_meta = resume_checkpoint
            
        except Exception as e:
            logger.warning(f"⚠ Could not resume from checkpoint: {e}")
            logger.info("Starting fresh from initial checkpoint instead")
            # Fall back to initial checkpoint loading below
            step = 0
            best_val_loss = float('inf')
    else:
        # Load optimizer/scheduler state from initial checkpoint if not resetting
        step = 0
        best_val_loss = float('inf')
        
        if not args.reset_optimizer and "optimizer" in checkpoint_meta:
            try:
                opt.load_state_dict(checkpoint_meta["optimizer"])
                print("✓ Optimizer state loaded from initial checkpoint")
            except Exception as e:
                print(f"⚠ Could not load optimizer state: {e}")
        
        if not args.reset_scheduler and "scheduler" in checkpoint_meta:
            try:
                scheduler.load_state_dict(checkpoint_meta["scheduler"])
                print("✓ Scheduler state loaded from initial checkpoint")
            except Exception as e:
                print(f"⚠ Could not load scheduler state: {e}")
        
        if not args.reset_step and "step" in checkpoint_meta:
            step = checkpoint_meta["step"]
            print(f"✓ Resuming from step {step}")
        else:
            step = 0
            print("Starting from step 0")
        
        if "best_val_loss" in checkpoint_meta:
            best_val_loss = checkpoint_meta["best_val_loss"]
            print(f"✓ Previous best validation loss: {best_val_loss:.4f}")
    
    # Setup mixed precision
    use_amp = cfg.get("use_amp", True) and device == "cuda"
    scaler = GradScaler('cuda') if use_amp else None
    if use_amp:
        if not args.reset_optimizer and "scaler" in checkpoint_meta and scaler is not None:
            try:
                scaler.load_state_dict(checkpoint_meta["scaler"])
                print("✓ Scaler state loaded")
            except Exception as e:
                print(f"⚠ Could not load scaler state: {e}")
        print("Mixed precision training (AMP) enabled")
    
    # Gradient settings
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    accumulation_steps = cfg.get("gradient_accumulation_steps", 1)
    if accumulation_steps > 1:
        print(f"Gradient accumulation: {accumulation_steps} steps")
    
    # Logger already initialized above for resumption
    train_size = len(train_dl.dataset)
    val_size = len(val_dl.dataset)
    logger.training_start(max_steps, train_size, val_size)
    logger.info(f"Post-training {model_type} from checkpoint: {args.checkpoint}")
    logger.info(f"New dataset: {args.new_dataset} ({train_size + val_size:,} samples)")
    if args.reset_optimizer:
        logger.info("Optimizer state reset (fine-tuning mode)")
    if args.reset_scheduler:
        logger.info("Scheduler state reset (fine-tuning mode)")
    
    # Training loop
    model.train()
    if head:
        head.train()
    if rvq:
        rvq.train()
    if img_proj:
        img_proj.train()
    if text_proj:
        text_proj.train()
    if text_embed_encode:
        text_embed_encode[0].train()
    
    if model_type == "audio_enc":
        char_to_idx = getattr(head, "char_to_idx", None)
        if char_to_idx is None:
            raise ValueError("Audio encoder head is missing char_to_idx mapping for post-training.")
        max_text_len = getattr(head, "max_text_len", cfg.get("max_text_len", 64))
        unk_idx = getattr(head, "unk_idx", char_to_idx.get('<UNK>', len(char_to_idx)))

        def encode_texts(text_batch):
            encoded = []
            lengths = []
            for t in text_batch:
                ids = [char_to_idx.get(c, unk_idx) for c in t[:max_text_len]]
                if not ids:
                    ids = [unk_idx]
                encoded.append(torch.tensor(ids, dtype=torch.long))
                lengths.append(len(ids))
            if encoded:
                encoded = torch.cat(encoded)
            else:
                encoded = torch.empty(0, dtype=torch.long)
            return encoded, torch.tensor(lengths, dtype=torch.long)
    else:
        encode_texts = None
    
    max_epochs = cfg.get("max_epochs", 9999)
    print_freq = cfg.get("print_freq", 1)
    checkpoint_freq = cfg.get("checkpoint_freq", 500)
    val_freq = cfg.get("val_freq", 200)
    
    steps_per_epoch = len(train_dl)
    start_epoch = step // steps_per_epoch if step > 0 else 0
    initial_step = step
    
    for epoch in range(start_epoch, max_epochs):
        logger.epoch_start(epoch)
        for batch_idx, batch in enumerate(tqdm(train_dl, desc=f"epoch{epoch}")):
            # Skip batches if resuming mid-epoch
            if epoch == start_epoch and initial_step > 0:
                current_batch_step = epoch * steps_per_epoch + batch_idx
                if current_batch_step < initial_step:
                    continue
            
            # Prepare batch based on model type
            if model_type == "thinker":
                x, y = batch
                x, y = x.to(device), y.to(device)
            elif model_type == "audio_enc":
                mels, texts = batch
                mels = mels.to(device)
                targets, target_lens = encode_texts(texts)
                targets = targets.to(device)
                target_lens = target_lens.to(device)
            elif model_type == "vision":
                imgs, captions = batch
                imgs = imgs.to(device)
            elif model_type == "talker":
                mels = batch.to(device)
                codes = rvq.encode(mels)
                prev = torch.roll(codes, 1, dims=1)
                prev[:, 0, :] = 0
            
            # Mark step begin for CUDAGraphs optimization
            if device == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            
            # Forward pass
            try:
                if use_amp:
                    with autocast(device_type='cuda'):
                        if model_type == "thinker":
                            logits = model(x)
                            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                            del logits
                        elif model_type == "audio_enc":
                            out = model(mels)
                            logits = head(out)
                            log_prob = logits.log_softmax(-1).transpose(0, 1)
                            inp_lens = torch.full((log_prob.size(1),), log_prob.size(0), dtype=torch.long, device=log_prob.device)
                            loss = loss_fn(log_prob, targets, inp_lens, target_lens)
                            del out, logits, log_prob
                        elif model_type == "vision":
                            cls, _ = model(imgs)
                            img_feat = img_proj(cls.squeeze(1))
                            text_feats = torch.stack([text_embed_encode[1](cap) for cap in captions]).to(device)
                            text_feat = text_proj(text_feats)
                            loss = loss_fn(img_feat, text_feat)
                            del cls, img_feat, text_feats, text_feat
                        elif model_type == "talker":
                            base_logits, res_logits = model(prev, use_cache=False)
                            loss = (
                                loss_fn(base_logits.reshape(-1, base_logits.size(-1)), codes[:, :, 0].reshape(-1)) +
                                loss_fn(res_logits.reshape(-1, res_logits.size(-1)), codes[:, :, 1].reshape(-1))
                            )
                            del base_logits, res_logits
                else:
                    if model_type == "thinker":
                        logits = model(x)
                        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                        del logits
                    elif model_type == "audio_enc":
                        out = model(mels)
                        logits = head(out)
                        log_prob = logits.log_softmax(-1).transpose(0, 1)
                        inp_lens = torch.full((log_prob.size(1),), log_prob.size(0), dtype=torch.long, device=log_prob.device)
                        loss = loss_fn(log_prob, targets, inp_lens, target_lens)
                        del out, logits, log_prob
                    elif model_type == "vision":
                        cls, _ = model(imgs)
                        img_feat = img_proj(cls.squeeze(1))
                        text_feats = torch.stack([text_embed_encode[1](cap) for cap in captions]).to(device)
                        text_feat = text_proj(text_feats)
                        loss = loss_fn(img_feat, text_feat)
                        del cls, img_feat, text_feats, text_feat
                    elif model_type == "talker":
                        base_logits, res_logits = model(prev, use_cache=False)
                        loss = (
                            loss_fn(base_logits.reshape(-1, base_logits.size(-1)), codes[:, :, 0].reshape(-1)) +
                            loss_fn(res_logits.reshape(-1, res_logits.size(-1)), codes[:, :, 1].reshape(-1))
                        )
                        del base_logits, res_logits
            except RuntimeError as e:
                error_msg = str(e)
                if ("NaN detected in attention probabilities after softmax" in error_msg or "Numerical instability" in error_msg) and model_type == "thinker":
                    logger.error(f"Step {step}: {e}")
                    logger.error("Reloading from last checkpoint...")
                    # Reload from last checkpoint
                    checkpoint_prefix = f"{model_type}_post_step_"
                    reloaded_step, reloaded_best_val_loss = reload_from_last_checkpoint(
                        cfg["save_dir"], checkpoint_prefix, device, logger, model, opt, scheduler, scaler
                    )
                    if reloaded_step > 0:
                        step = reloaded_step
                        best_val_loss = reloaded_best_val_loss
                        # Recalculate start_epoch and initial_step for resuming
                        start_epoch = step // steps_per_epoch
                        initial_step = step
                        logger.info(f"Resuming from step {step}, epoch {start_epoch}")
                    opt.zero_grad()
                    if use_amp:
                        scaler.update()
                    continue
                else:
                    # Re-raise if it's a different error or different model type
                    raise
            
            # Scale loss for gradient accumulation
            loss_scaled = loss / accumulation_steps
            
            # Backward pass
            if use_amp:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()
            
            # Detach loss for logging (free computation graph)
            loss_val = loss.detach()
            del loss
            
            # Gradient accumulation
            if (step + 1) % accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(opt)
                
                unscaled_loss = loss_val * accumulation_steps
                try:
                    validate_loss(unscaled_loss, min_loss=-1e6, max_loss=1e6)
                except RuntimeError as e:
                    logger.error(f"Step {step}: {e}")
                    opt.zero_grad()
                    if use_amp:
                        scaler.update()
                    continue
                
                try:
                    grad_norm, is_exploded = check_gradient_explosion(model, max_grad_norm=100.0, raise_on_error=False)
                    if is_exploded:
                        logger.error(f"Step {step}: Gradient explosion. Skipping batch.")
                        opt.zero_grad()
                        if use_amp:
                            scaler.update()
                        continue
                except RuntimeError as e:
                    logger.error(f"Step {step}: {e}")
                    opt.zero_grad()
                    if use_amp:
                        scaler.update()
                    continue
                
                if use_amp:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    opt.step()
                
                scheduler.step()
                
                # Check for NaN
                if hasattr(model, 'check_weights_stability'):
                    has_nan, has_inf, nan_count, inf_count = model.check_weights_stability()
                    if has_nan or has_inf:
                        logger.error(f"Step {step}: Model weights corrupted. Skipping.")
                        opt.zero_grad()
                        continue
                
                opt.zero_grad()
                
                # Periodic memory cleanup
                if step % 100 == 0 and device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
            else:
                unscaled_loss = loss_val * accumulation_steps
                try:
                    validate_loss(unscaled_loss, min_loss=-1e6, max_loss=1e6)
                except RuntimeError as e:
                    logger.error(f"Step {step}: {e}")
                    continue
            
            step += 1
            
            # Logging
            if step % print_freq == 0:
                current_lr = scheduler.get_last_lr()[0]
                unscaled_loss = loss_val * accumulation_steps
                logger.train_step(step, float(unscaled_loss), current_lr, epoch)
            
            # Checkpointing
            if step % checkpoint_freq == 0 and step > 0:
                checkpoint_path = os.path.join(cfg["save_dir"], f"{model_type}_post_step_{step}.pt")
                checkpoint_data = {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "source_checkpoint": args.checkpoint,
                    "post_training": True
                }
                if model_type == "thinker":
                    checkpoint_data["model"] = model.state_dict()
                elif model_type == "audio_enc":
                    checkpoint_data["enc"] = model.state_dict()
                    checkpoint_data["head"] = head.state_dict()
                elif model_type == "vision":
                    checkpoint_data["vit"] = model.state_dict()
                    checkpoint_data["img_proj"] = img_proj.state_dict()
                    checkpoint_data["text_proj"] = text_proj.state_dict()
                    checkpoint_data["text_embed"] = text_embed_encode[0].state_dict()
                elif model_type == "talker":
                    checkpoint_data["talker"] = model.state_dict()
                    checkpoint_data["rvq"] = rvq.state_dict()
                
                if scaler is not None:
                    checkpoint_data["scaler"] = scaler.state_dict()
                torch.save(checkpoint_data, checkpoint_path)
                logger.checkpoint(step, checkpoint_path)
                
                # Clean up old checkpoints (keep only last one + best)
                prefix_map = {
                    "thinker": "thinker_step_",
                    "audio_enc": "audio_enc_step_",
                    "vision": "vision_step_",
                    "talker": "talker_step_"
                }
                cleanup_old_checkpoints(cfg["save_dir"], prefix_map[model_type], keep_last_n=1)
            
            if step % val_freq == 0 and step > 0:
                model.eval()
                if head:
                    head.eval()
                if rvq:
                    rvq.eval()
                if img_proj:
                    img_proj.eval()
                if text_proj:
                    text_proj.eval()
                if text_embed_encode:
                    text_embed_encode[0].eval()
                
                val_loss_sum = 0.0
                val_batches = 0
                with torch.no_grad():
                    for val_batch in val_dl:
                        if model_type == "thinker":
                            val_x, val_y = val_batch
                            val_x, val_y = val_x.to(device), val_y.to(device)
                            val_logits = model(val_x)
                            val_loss = loss_fn(val_logits.view(-1, val_logits.size(-1)), val_y.view(-1))
                        elif model_type == "audio_enc":
                            val_mels, val_texts = val_batch
                            val_mels = val_mels.to(device)
                            val_targets, val_target_lens = encode_texts(val_texts)
                            val_targets = val_targets.to(device)
                            val_target_lens = val_target_lens.to(device)
                            val_out = model(val_mels)
                            val_logits = head(val_out)
                            val_log_prob = val_logits.log_softmax(-1).transpose(0, 1)
                            val_inp_lens = torch.full((val_log_prob.size(1),), val_log_prob.size(0), dtype=torch.long, device=val_log_prob.device)
                            val_loss = loss_fn(val_log_prob, val_targets, val_inp_lens, val_target_lens)
                        elif model_type == "vision":
                            val_imgs, val_caps = val_batch
                            val_imgs = val_imgs.to(device)
                            val_cls, _ = model(val_imgs)
                            val_img_feat = img_proj(val_cls.squeeze(1))
                            val_img_feat = val_img_feat / val_img_feat.norm(dim=-1, keepdim=True)
                            val_text_feats = torch.stack([text_embed_encode[1](cap) for cap in val_caps]).to(device)
                            val_text_feat = text_proj(val_text_feats)
                            val_text_feat = val_text_feat / val_text_feat.norm(dim=-1, keepdim=True)
                            temperature = cfg.get("temperature", 0.07)
                            val_logits = torch.matmul(val_img_feat, val_text_feat.t()) / temperature
                            val_labels = torch.arange(val_logits.size(0), device=device)
                            val_loss = nn.CrossEntropyLoss()(val_logits, val_labels)
                        elif model_type == "talker":
                            val_mels = val_batch.to(device)
                            val_codes = rvq.encode(val_mels)
                            val_prev = torch.roll(val_codes, 1, dims=1)
                            val_prev[:, 0, :] = 0
                            val_base_logits, val_res_logits = model(val_prev, use_cache=False)
                            val_loss = (
                                loss_fn(val_base_logits.reshape(-1, val_base_logits.size(-1)), val_codes[:, :, 0].reshape(-1)) +
                                loss_fn(val_res_logits.reshape(-1, val_res_logits.size(-1)), val_codes[:, :, 1].reshape(-1))
                            )
                        else:
                            val_loss = torch.tensor(0.0, device=device)
                        
                        try:
                            validate_loss(val_loss, min_loss=-1e6, max_loss=1e6)
                        except RuntimeError as e:
                            logger.warning(f"Step {step}: Validation error: {e}")
                            continue
                        
                        val_loss_sum += float(val_loss.detach())
                        val_batches += 1
                        # Free validation tensors based on model type
                        if model_type == "thinker":
                            del val_logits, val_loss
                        elif model_type == "audio_enc":
                            del val_out, val_logits, val_log_prob, val_loss
                        elif model_type == "vision":
                            del val_cls, val_img_feat, val_text_feats, val_text_feat, val_logits, val_loss
                        elif model_type == "talker":
                            del val_base_logits, val_res_logits, val_codes, val_prev, val_loss
                        if val_batches >= 20:
                            break
                
                avg_val_loss = val_loss_sum / max(val_batches, 1)
                logger.val_step(step, avg_val_loss, epoch)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_path = os.path.join(cfg["save_dir"], f"{model_type}_post_best.pt")
                    best_checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step,
                        "best_val_loss": best_val_loss,
                        "source_checkpoint": args.checkpoint,
                        "post_training": True
                    }
                    if model_type == "thinker":
                        best_checkpoint["model"] = model.state_dict()
                    elif model_type == "audio_enc":
                        best_checkpoint["enc"] = model.state_dict()
                        best_checkpoint["head"] = head.state_dict()
                    elif model_type == "vision":
                        best_checkpoint["vit"] = model.state_dict()
                        best_checkpoint["img_proj"] = img_proj.state_dict()
                        best_checkpoint["text_proj"] = text_proj.state_dict()
                        best_checkpoint["text_embed"] = text_embed_encode[0].state_dict()
                    elif model_type == "talker":
                        best_checkpoint["talker"] = model.state_dict()
                        best_checkpoint["rvq"] = rvq.state_dict()
                    if scaler is not None:
                        best_checkpoint["scaler"] = scaler.state_dict()
                    torch.save(best_checkpoint, best_path)
                    logger.checkpoint(step, best_path, is_best=True)
                
                model.train()
                if head:
                    head.train()
                if rvq:
                    rvq.train()
                if img_proj:
                    img_proj.train()
                if text_proj:
                    text_proj.train()
                if text_embed_encode:
                    text_embed_encode[0].train()
                model.train()
            
            # Stop at max_steps
            if step >= max_steps:
                final_path = os.path.join(cfg["save_dir"], f"{model_type}_post_final.pt")
                checkpoint_data = {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "source_checkpoint": args.checkpoint,
                    "post_training": True
                }
                torch.save(checkpoint_data, final_path)
                logger.info(f"Final model saved to {final_path}")
                logger.training_end(step)
                return

def main():
    parser = argparse.ArgumentParser(description="Post-training: Continue training from checkpoint with new dataset")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file (.pt)")
    parser.add_argument("--new_dataset", required=True, help="Path to new dataset file")
    parser.add_argument("--model_type", choices=["thinker", "audio_enc", "vision", "talker"],
                       help="Model type (auto-detected from config if not specified)")
    parser.add_argument("--reset_optimizer", action="store_true", help="Reset optimizer state")
    parser.add_argument("--reset_scheduler", action="store_true", help="Reset scheduler state")
    parser.add_argument("--reset_step", action="store_true", help="Reset step counter to 0")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--wd", type=float, help="Override weight decay")
    parser.add_argument("--warmup_steps", type=int, help="Override warmup steps")
    parser.add_argument("--max_steps", type=int, help="Override max steps")
    parser.add_argument("--save_dir", type=str, help="Override save directory")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    
    # Override config
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.wd is not None:
        cfg["wd"] = args.wd
    if args.warmup_steps is not None:
        cfg["warmup_steps"] = args.warmup_steps
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.save_dir is not None:
        cfg["save_dir"] = args.save_dir
    
    # Detect model type
    model_type = args.model_type or detect_model_type(cfg)
    print(f"Detected model type: {model_type}")
    
    # Set random seed
    seed = cfg.get("seed", 42)
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Override save_dir for post-training (use dedicated post_training directory)
    original_save_dir = cfg["save_dir"]
    cfg["save_dir"] = "checkpoints/post_training"
    os.makedirs(cfg["save_dir"], exist_ok=True)
    print(f"Post-training checkpoints will be saved to: {cfg['save_dir']}")
    print(f"Original model directory: {original_save_dir}")
    
    # Setup model based on type
    if model_type == "thinker":
        model, opt, loss_fn, train_dl, val_dl, checkpoint_meta, _, head, rvq, img_proj, text_embed_encode = \
            train_thinker(cfg, args.checkpoint, args.new_dataset, args, device)
    elif model_type == "audio_enc":
        model, opt, loss_fn, train_dl, val_dl, checkpoint_meta, _, head, rvq, img_proj, text_embed_encode = \
            train_audio_enc(cfg, args.checkpoint, args.new_dataset, args, device)
    elif model_type == "vision":
        model, opt, loss_fn, train_dl, val_dl, checkpoint_meta, _, head, rvq, img_proj, text_embed_encode = \
            train_vision(cfg, args.checkpoint, args.new_dataset, args, device)
    elif model_type == "talker":
        model, opt, loss_fn, train_dl, val_dl, checkpoint_meta, _, head, rvq, img_proj, text_embed_encode = \
            train_talker(cfg, args.checkpoint, args.new_dataset, args, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Run training loop
    run_training_loop(model, opt, loss_fn, train_dl, val_dl, checkpoint_meta, model_type,
                     head, rvq, img_proj, text_proj, text_embed_encode, cfg, args, device)

if __name__ == "__main__":
    main()
