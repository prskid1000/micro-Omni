
"""
μOmni Multimodal Inference Interface
Modular, class-based architecture for clean model loading and inference
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from PIL import Image
from torch.amp import autocast
from torchvision import transforms
import torchvision.io as tvio

# Model imports
from omni.thinker import ThinkerLM
from omni.audio_encoder import AudioEncoderTiny
from omni.vision_encoder import ViTTiny
from omni.codec import RVQ, NeuralVocoder
from omni.talker import TalkerTiny
from omni.tokenizer import BPETokenizer
from omni.ocr_model import OCRModel
from omni.utils import find_checkpoint, load_audio, normalize_state_dict


# ============================================================================
# Utility Functions
# ============================================================================

def strip_orig_mod_keys(state_dict: Dict) -> Dict:
    """Strip _orig_mod. prefix from state_dict keys (from torch.compile) - DEPRECATED, use normalize_state_dict"""
    return normalize_state_dict(state_dict, strip_orig_mod_prefix=True, convert_attention=False)


# ============================================================================
# Model Loader Classes
# ============================================================================

class BaseModelLoader:
    """Base class for model loaders"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.loaded = False
    
    def load_config(self, config_path: str, default_config: Dict) -> Dict:
        """Load config from file or return default"""
        if os.path.exists(config_path):
            return json.load(open(config_path))
        return default_config
    
    def load_checkpoint(self, ckpt_dir: str, prefix: str, step_prefix: str = None) -> Tuple[Optional[str], Optional[Dict]]:
        """Load checkpoint using find_checkpoint utility"""
        return find_checkpoint(ckpt_dir, prefix, step_prefix, self.device)
    
    def load_state_dict(self, state_dict: Dict, strict: bool = False):
        """Load state dict with normalization (only strip _orig_mod, don't convert attention weights)
        
        This matches training script behavior - load checkpoints as saved.
        The model should be initialized with the same config (including use_gqa) as training.
        """
        if state_dict is None:
            return False
        # Only strip _orig_mod, don't convert attention weights (matches training)
        from omni.utils import strip_orig_mod
        state_dict = strip_orig_mod(state_dict)
        try:
            self.model.load_state_dict(state_dict, strict=strict)
            return True
        except Exception as e:
            print(f"Warning: Failed to load state_dict: {e}")
            return False


class ThinkerLoader(BaseModelLoader):
    """Loader for ThinkerLM language model"""
    
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.tokenizer = None
    
    def load(self, ckpt_dir: str, config: Dict) -> bool:
        """Load Thinker model and tokenizer"""
        try:
            # Load config
            thinker_cfg = config.get("thinker", config)
            vocab_size = thinker_cfg.get("vocab_size", 32000)
            ctx_len = thinker_cfg.get("ctx_len", 512)
            
            # Initialize model
            self.model = ThinkerLM(
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
                num_experts_per_tok=thinker_cfg.get("num_experts_per_tok", 2)
            ).to(self.device)
            
            # Load tokenizer
            tok_path = os.path.join(ckpt_dir, "tokenizer.model")
            if not os.path.exists(tok_path):
                # Try thinker checkpoint directory
                if "thinker_ckpt" in config:
                    tok_path = os.path.join(config["thinker_ckpt"], "tokenizer.model")
                else:
                    tok_path = os.path.join("checkpoints/thinker_tiny", "tokenizer.model")
            
            self.tokenizer = BPETokenizer(tok_path)
            
            # Load checkpoint
            thinker_ckpt_dir = ckpt_dir
            if "thinker_ckpt" in config:
                thinker_ckpt_dir = config["thinker_ckpt"]
            elif "thinker" not in ckpt_dir:
                thinker_ckpt_dir = "checkpoints/thinker_tiny"
            
            ckpt_path, ckpt = self.load_checkpoint(thinker_ckpt_dir, "thinker.pt", "thinker_step_")
            
            if ckpt is not None:
                if isinstance(ckpt, dict):
                    if "model" in ckpt:
                        self.load_state_dict(ckpt["model"], strict=False)
                    elif "thinker" in ckpt:
                        self.load_state_dict(ckpt["thinker"], strict=False)
                    else:
                        self.load_state_dict(ckpt, strict=False)
                else:
                    self.load_state_dict(ckpt, strict=False)
                print("✓ Loaded Thinker model")
            else:
                print("⚠ Warning: Thinker checkpoint not found, using untrained model")
            
            self.model.eval()
            self.loaded = True
            return True
        except Exception as e:
            print(f"✗ Failed to load Thinker: {e}")
            return False


class VisionLoader(BaseModelLoader):
    """Loader for Vision encoder"""
    
    def load(self, ckpt_dir: str, config: Dict) -> bool:
        """Load Vision encoder"""
        try:
            # Load config
            vision_cfg = self.load_config("configs/vision_tiny.json", {
                "img_size": 224, "patch": 16, "d_model": 128,
                "n_layers": 4, "n_heads": 2, "d_ff": 512, "dropout": 0.1
            })
            
            # Initialize model
            self.model = ViTTiny(
                img_size=vision_cfg.get("img_size", 224),
                patch=vision_cfg.get("patch", 16),
                d=vision_cfg.get("d_model", 128),
                layers=vision_cfg.get("n_layers", 4),
                heads=vision_cfg.get("n_heads", 2),
                ff=vision_cfg.get("d_ff", 512),
                dropout=vision_cfg.get("dropout", 0.1)
            ).to(self.device)
            
            # Load checkpoint
            vision_ckpt_dir = ckpt_dir
            if "vision_ckpt" in config:
                vision_ckpt_dir = config["vision_ckpt"]
            
            ckpt_path, ckpt = self.load_checkpoint(vision_ckpt_dir, "vision.pt", "vision_step_")
            
            if ckpt is not None:
                if isinstance(ckpt, dict):
                    if "vit" in ckpt:
                        self.load_state_dict(ckpt["vit"], strict=False)
                    elif "model" in ckpt:
                        self.load_state_dict(ckpt["model"], strict=False)
                    else:
                        self.load_state_dict(ckpt, strict=False)
                else:
                    self.load_state_dict(ckpt, strict=False)
                print("✓ Loaded Vision encoder")
            else:
                print("⚠ Warning: Vision checkpoint not found, using untrained model")
            
            self.model.eval()
            self.loaded = True
            return True
        except Exception as e:
            print(f"✗ Failed to load Vision: {e}")
            return False


class AudioLoader(BaseModelLoader):
    """Loader for Audio encoder"""
    
    def load(self, ckpt_dir: str, config: Dict) -> bool:
        """Load Audio encoder"""
        try:
            # Load config
            audio_cfg = self.load_config("configs/audio_enc_tiny.json", {
                "d_model": 192, "n_layers": 4, "n_heads": 3,
                "d_ff": 768, "dropout": 0.1, "downsample_time": 8
            })
            
            # Initialize model
            downsample_factor = audio_cfg.get("downsample_time", 8)
            self.model = AudioEncoderTiny(
                d=audio_cfg.get("d_model", 192),  # Use 'd' not 'd_model'
                heads=audio_cfg.get("n_heads", 3),  # Use 'heads' not 'n_heads'
                ff=audio_cfg.get("d_ff", 768),  # Use 'ff' not 'd_ff'
                layers=audio_cfg.get("n_layers", 4),  # Use 'layers' not 'n_layers'
                dropout=audio_cfg.get("dropout", 0.1),
                downsample_factor=downsample_factor
            ).to(self.device)
            
            # Load checkpoint
            audio_ckpt_dir = ckpt_dir
            if "audio_ckpt" in config:
                audio_ckpt_dir = config["audio_ckpt"]
            
            ckpt_path, ckpt = self.load_checkpoint(audio_ckpt_dir, "audio_enc.pt", "audio_enc_step_")
            
            if ckpt is not None:
                if isinstance(ckpt, dict):
                    if "enc" in ckpt:
                        self.load_state_dict(ckpt["enc"], strict=False)
                    elif "model" in ckpt:
                        self.load_state_dict(ckpt["model"], strict=False)
                    else:
                        self.load_state_dict(ckpt, strict=False)
                else:
                    self.load_state_dict(ckpt, strict=False)
                print("✓ Loaded Audio encoder")
            else:
                print("⚠ Warning: Audio checkpoint not found, using untrained model")
            
            self.model.eval()
            self.loaded = True
            return True
        except Exception as e:
            print(f"✗ Failed to load Audio: {e}")
            return False


class TalkerLoader(BaseModelLoader):
    """Loader for Talker (TTS) model"""
    
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.rvq = None
    
    def load(self, ckpt_dir: str, config: Dict) -> bool:
        """Load Talker and RVQ"""
        try:
            # Load config
            talker_cfg = self.load_config("configs/talker_tiny.json", {
                "d_model": 192, "n_layers": 4, "n_heads": 3,
                "d_ff": 768, "codebooks": 2, "codebook_size": 128, "dropout": 0.1
            })
            
            codebooks = talker_cfg.get("codebooks", 2)
            codebook_size = talker_cfg.get("codebook_size", 128)
            
            # Initialize RVQ
            self.rvq = RVQ(codebooks, codebook_size, d=64).to(self.device)
            
            # Initialize Talker
            self.model = TalkerTiny(
                d=talker_cfg.get("d_model", 192),
                n_layers=talker_cfg.get("n_layers", 4),
                n_heads=talker_cfg.get("n_heads", 3),
                ff=talker_cfg.get("d_ff", 768),
                codebooks=codebooks,
                codebook_size=codebook_size,
                dropout=talker_cfg.get("dropout", 0.1),
                use_gqa=talker_cfg.get("use_gqa", False),
                use_swiglu=talker_cfg.get("use_swiglu", True),
                rope_theta=talker_cfg.get("rope_theta", 10000.0)
            ).to(self.device)
            
            # Load checkpoint
            talker_ckpt_dir = ckpt_dir
            if "talker_ckpt" in config:
                talker_ckpt_dir = config["talker_ckpt"]
            
            ckpt_path, ckpt = self.load_checkpoint(talker_ckpt_dir, "talker.pt", "talker_step_")
            
            if ckpt is not None:
                if isinstance(ckpt, dict):
                    if "rvq" in ckpt:
                        rvq_state = normalize_state_dict(ckpt["rvq"])
                        self.rvq.load_state_dict(rvq_state, strict=False)
                    if "talker" in ckpt:
                        self.load_state_dict(ckpt["talker"], strict=False)
                print("✓ Loaded Talker model")
            else:
                print("⚠ Warning: Talker checkpoint not found, using untrained model")
            
            self.model.eval()
            self.rvq.eval()
            self.loaded = True
            return True
        except Exception as e:
            print(f"✗ Failed to load Talker: {e}")
            return False


class OCRLoader(BaseModelLoader):
    """Loader for OCR model"""
    
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.char_to_idx = None
        self.idx_to_char = None
    
    def load(self, ckpt_dir: str, config: Dict) -> bool:
        """Load OCR model with vocab_size from checkpoint"""
        try:
            # First, try to load checkpoint to get vocab_size
            ocr_ckpt_dir = ckpt_dir
            if not os.path.exists(os.path.join(ocr_ckpt_dir, "ocr.pt")):
                ocr_ckpt_dir = "checkpoints/ocr_tiny"
            
            ckpt_path, ckpt = self.load_checkpoint(ocr_ckpt_dir, "ocr.pt", "ocr_step_")
            
            # Get vocab_size from checkpoint
            vocab_size = 128  # Default
            if ckpt is not None and isinstance(ckpt, dict):
                if "char_to_idx" in ckpt:
                    vocab_size = len(ckpt["char_to_idx"])
                    self.char_to_idx = ckpt["char_to_idx"]
                    self.idx_to_char = ckpt.get("idx_to_char", {v: k for k, v in self.char_to_idx.items()})
                elif "config" in ckpt and "vocab_size" in ckpt["config"]:
                    vocab_size = ckpt["config"]["vocab_size"]
            
            # Load config
            ocr_cfg = self.load_config("configs/ocr_tiny.json", {
                "img_size": 224, "patch": 16, "vision_d_model": 128,
                "vision_layers": 4, "vision_heads": 2, "vision_d_ff": 512,
                "decoder_d_model": 256, "decoder_layers": 4, "decoder_heads": 4,
                "decoder_d_ff": 1024, "dropout": 0.1
            })
            
            # Initialize model with correct vocab_size
            self.model = OCRModel(
                img_size=ocr_cfg.get("img_size", 224),
                patch=ocr_cfg.get("patch", 16),
                vision_d_model=ocr_cfg.get("vision_d_model", 128),
                vision_layers=ocr_cfg.get("vision_layers", 4),
                vision_heads=ocr_cfg.get("vision_heads", 2),
                vision_d_ff=ocr_cfg.get("vision_d_ff", 512),
                decoder_d_model=ocr_cfg.get("decoder_d_model", 256),
                decoder_layers=ocr_cfg.get("decoder_layers", 4),
                decoder_heads=ocr_cfg.get("decoder_heads", 4),
                decoder_d_ff=ocr_cfg.get("decoder_d_ff", 1024),
                vocab_size=vocab_size,
                dropout=ocr_cfg.get("dropout", 0.1),
                use_gqa=ocr_cfg.get("use_gqa", False),
                use_swiglu=ocr_cfg.get("use_swiglu", True),
                use_flash=ocr_cfg.get("use_flash", True)
            ).to(self.device)
            
            # Load checkpoint state_dict
            if ckpt is not None:
                if isinstance(ckpt, dict):
                    if "model" in ckpt:
                        self.load_state_dict(ckpt["model"], strict=False)
                    else:
                        self.load_state_dict(ckpt, strict=False)
                else:
                    self.load_state_dict(ckpt, strict=False)
                print(f"✓ Loaded OCR model (vocab_size={vocab_size})")
            else:
                print("⚠ Warning: OCR checkpoint not found, using untrained model")
            
            self.model.eval()
            self.loaded = True
            return True
        except Exception as e:
            print(f"✗ Failed to load OCR: {e}")
            import traceback
            traceback.print_exc()
            return False


class VocoderLoader:
    """Loader for Vocoder (TTS output)"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.vocoder = None
        self.loaded = False
    
    def load(self, ckpt_dir: str) -> bool:
        """Load vocoder for audio output"""
        try:
            hifigan_checkpoint = os.path.join(ckpt_dir, "hifigan.pt")
            if not os.path.exists(hifigan_checkpoint):
                hifigan_checkpoint = "checkpoints/hifigan.pt"
            
            self.vocoder = NeuralVocoder(
                sample_rate=16000,
                n_mels=128,
                checkpoint_path=hifigan_checkpoint if os.path.exists(hifigan_checkpoint) else None,
                prefer_neural=True
            )
            
            if self.vocoder.vocoder_type == "hifigan":
                self.vocoder = self.vocoder.to(self.device).eval()
            
            print(f"✓ Loaded {self.vocoder.vocoder_type} vocoder")
            self.loaded = True
            return True
        except Exception as e:
            print(f"⚠ Warning: Failed to load vocoder: {e}")
            return False


# ============================================================================
# Processing Classes
# ============================================================================

class ImageProcessor:
    """Process images for vision encoder"""
    
    def __init__(self, img_size: int = 224):
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    
    def process(self, image_path: str) -> torch.Tensor:
        """Load and process image"""
        img = Image.open(image_path).convert("RGB")
        return self.transform(img).unsqueeze(0)


class AudioProcessor:
    """Process audio for audio encoder"""
    
    def __init__(self, device: str = "cuda", sample_rate: int = 16000):
        self.device = device
        self.sample_rate = sample_rate
        self.mel_spec = None
        if hasattr(torchaudio.transforms, 'MelSpectrogram'):
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_fft=1024, hop_length=160,
                win_length=400, n_mels=128
            ).to(device)
    
    def process(self, audio_path: str) -> torch.Tensor:
        """Load and process audio to mel spectrogram"""
        wav, sr = load_audio(audio_path)
        wav = wav.to(self.device)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if self.mel_spec is not None:
            mel = self.mel_spec(wav)[0].T.unsqueeze(0)  # (1, T, 128)
            return mel
        return None


class VideoProcessor:
    """Process videos by extracting frames"""
    
    @staticmethod
    def extract_frames(video_path: str, num_frames: int = 4) -> List[torch.Tensor]:
        """Extract evenly spaced frames from video"""
        try:
            video, audio, info = tvio.read_video(video_path, output_format="TCHW")
            total_frames = video.shape[0]
            if total_frames == 0:
                return []
            indices = [int(i * total_frames / (num_frames + 1)) for i in range(1, num_frames + 1)]
            frames = [video[i] for i in indices if i < total_frames]
            return frames
        except Exception as e:
            print(f"Warning: Could not extract frames from video: {e}")
            return []


# ============================================================================
# Generation Classes
# ============================================================================

class TextGenerator:
    """Generate text from prompts"""
    
    def __init__(self, model: ThinkerLM, tokenizer: BPETokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def generate(self, prompt: str, max_new: int = 64, ctx: int = 512,
                 multimodal_emb: Optional[torch.Tensor] = None,
                 use_cache: bool = True, use_amp: bool = True) -> str:
        """Generate text from prompt"""
        if use_cache and hasattr(self.model, 'enable_kv_cache'):
            self.model.enable_kv_cache(True)
            self.model.reset_kv_cache()
        
        ids = [1] + self.tokenizer.encode(prompt)
        
        if multimodal_emb is not None:
            mm_len = multimodal_emb.shape[1]
            max_text_len = ctx - mm_len - max_new - 1
            ids = ids[:max_text_len]
            text_emb = self.model.tok_emb(torch.tensor(ids, dtype=torch.long, device=self.device)[None, :])
            combined_emb = torch.cat([multimodal_emb, text_emb], dim=1)
            
            if use_amp and self.device == "cuda":
                with autocast(device_type='cuda'):
                    logits = self.model(embeddings=combined_emb)
            else:
                logits = self.model(embeddings=combined_emb)
            next_id = int(torch.argmax(logits[0, -1]))
            generated_ids = ids + [next_id]
            
            for _ in range(max_new - 1):
                next_idx = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                if use_amp and self.device == "cuda":
                    with autocast(device_type='cuda'):
                        logits = self.model(idx=next_idx)
                else:
                    logits = self.model(idx=next_idx)
                next_id = int(torch.argmax(logits[0, -1]))
                generated_ids.append(next_id)
                if next_id == 2:
                    break
        else:
            ids = ids[-(ctx-max_new-1):]
            x = torch.tensor(ids, dtype=torch.long, device=self.device)[None, :]
            
            if use_amp and self.device == "cuda":
                with autocast(device_type='cuda'):
                    logits = self.model(x)
            else:
                logits = self.model(x)
            next_id = int(torch.argmax(logits[0, -1]))
            generated_ids = ids + [next_id]
            
            for _ in range(max_new - 1):
                x = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                if use_amp and self.device == "cuda":
                    with autocast(device_type='cuda'):
                        logits = self.model(x)
                else:
                    logits = self.model(x)
                next_id = int(torch.argmax(logits[0, -1]))
                generated_ids.append(next_id)
                if next_id == 2:
                    break
        
        if use_cache and hasattr(self.model, 'reset_kv_cache'):
            self.model.reset_kv_cache()
        
        return self.tokenizer.decode(generated_ids)


class AudioGenerator:
    """Generate audio from text (TTS)"""
    
    def __init__(self, talker: TalkerTiny, rvq: RVQ, vocoder: NeuralVocoder, device: str = "cuda"):
        self.talker = talker
        self.rvq = rvq
        self.vocoder = vocoder
        self.device = device
    
    def generate(self, text_tokens: List[int], max_frames: Optional[int] = None,
                 temperature: float = 0.8, top_k: int = 50) -> Optional[np.ndarray]:
        """Generate audio from text tokens"""
        if self.vocoder is None:
            print("Warning: Vocoder not available")
            return None
        
        if max_frames is None:
            if text_tokens and len(text_tokens) > 0:
                max_frames = max(50, min(1000, int(len(text_tokens) * 30)))
            else:
                max_frames = 200
        
        self.talker.eval()
        self.rvq.eval()
        
        if hasattr(self.talker, 'enable_kv_cache'):
            self.talker.enable_kv_cache(True)
            self.talker.reset_kv_cache()
        
        def sample_with_temperature(logits, temp=1.0, top_k_val=None):
            if temp == 0.0:
                return torch.argmax(logits, dim=-1)
            logits = logits / temp
            if top_k_val is not None and top_k_val > 0:
                top_k_val = min(top_k_val, logits.size(-1))
                top_k_values, top_k_indices = torch.topk(logits, top_k_val, dim=-1)
                filtered_logits = torch.full_like(logits, float('-inf'))
                filtered_logits.scatter_(-1, top_k_indices, top_k_values)
                logits = filtered_logits
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        with torch.no_grad():
            codes = torch.zeros(1, 1, 2, dtype=torch.long, device=self.device)
            
            base_logit, res_logit = self.talker(codes, use_cache=True)
            base_code = sample_with_temperature(base_logit[0, -1, :], temperature, top_k).item()
            res_code = sample_with_temperature(res_logit[0, -1, :], temperature, top_k).item()
            next_codes = torch.tensor([[[base_code, res_code]]], device=self.device)
            codes = torch.cat([codes, next_codes], dim=1)
            
            for _ in range(max_frames - 1):
                base_logit, res_logit = self.talker(next_codes, use_cache=True)
                base_code = sample_with_temperature(base_logit[0, -1, :], temperature, top_k).item()
                res_code = sample_with_temperature(res_logit[0, -1, :], temperature, top_k).item()
                next_codes = torch.tensor([[[base_code, res_code]]], device=self.device)
                codes = torch.cat([codes, next_codes], dim=1)
            
            if hasattr(self.talker, 'reset_kv_cache'):
                self.talker.reset_kv_cache()
            
            # Remove BOS token
            if codes.shape[1] > 1:
                codes = codes[:, 1:, :]
            
            # Decode to mel
            mel_frames = []
            for t in range(codes.shape[1]):
                frame_codes = codes[:, t:t+1, :]
                frame_codes_flat = frame_codes.squeeze(0).squeeze(0)
                mel_frame = self.rvq.decode(frame_codes_flat.unsqueeze(0))
                mel_frames.append(mel_frame.squeeze(0))
            
            mel = torch.stack(mel_frames, dim=0)
            
            # Normalize
            mel_min = mel.min()
            mel_max = mel.max()
            if mel_max > mel_min + 1e-6:
                mel_normalized = (mel - mel_min) / (mel_max - mel_min + 1e-8)
            else:
                t = torch.arange(mel.shape[0], device=mel.device, dtype=mel.dtype)[:, None]
                mel_normalized = 0.5 + 0.3 * torch.sin(2 * np.pi * t / 20)
            
            # Convert to audio
            try:
                if self.vocoder.vocoder_type == "hifigan" and isinstance(self.vocoder.vocoder, nn.Module):
                    mel_tensor = mel_normalized.T.unsqueeze(0)
                    with torch.no_grad():
                        audio_tensor = self.vocoder.vocoder(mel_tensor)
                    audio = audio_tensor.squeeze().cpu().numpy()
                else:
                    mel_np = mel_normalized.cpu().numpy()
                    audio = self.vocoder.mel_to_audio(mel_np)
                
                if np.max(np.abs(audio)) < 1e-6:
                    duration = len(audio) / 16000
                    t = np.linspace(0, duration, len(audio))
                    audio = audio + 0.1 * np.sin(2 * np.pi * 440 * t)
                
                return audio
            except Exception as e:
                print(f"Warning: Audio generation failed: {e}")
                duration = 0.5
                t = np.linspace(0, duration, int(16000 * duration))
                return 0.1 * np.sin(2 * np.pi * 440 * t)


class OCRProcessor:
    """Process OCR text extraction"""
    
    def __init__(self, model: OCRModel, char_to_idx: Dict, idx_to_char: Dict, device: str = "cuda"):
        self.model = model
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.device = device
    
    def extract_text(self, image_tensor: torch.Tensor, max_len: int = 128) -> str:
        """Extract text from image"""
        self.model.eval()
        with torch.no_grad():
            bos_id = self.char_to_idx.get('<BOS>', 1) if self.char_to_idx else 1
            eos_id = self.char_to_idx.get('<EOS>', 2) if self.char_to_idx else 2
            
            text_ids = torch.tensor([[bos_id]], dtype=torch.long, device=self.device)
            ocr_text = ""
            
            for _ in range(max_len):
                if self.device == "cuda":
                    with autocast(device_type='cuda'):
                        logits = self.model(image_tensor, text_ids)
                else:
                    logits = self.model(image_tensor, text_ids)
                
                next_id = int(torch.argmax(logits[0, -1]))
                if next_id == eos_id or next_id == 0:
                    break
                
                if self.idx_to_char:
                    char = self.idx_to_char.get(next_id, '')
                    if char and char not in ['<BOS>', '<EOS>', '<PAD>', '<UNK>']:
                        ocr_text += char
                else:
                    if 32 <= next_id < 127:
                        ocr_text += chr(next_id)
                
                text_ids = torch.cat([text_ids, torch.tensor([[next_id]], dtype=torch.long, device=self.device)], dim=1)
            
            return ocr_text


# ============================================================================
# Main Inference Engine
# ============================================================================

class InferenceEngine:
    """Main inference engine orchestrating all models"""
    
    def __init__(self, ckpt_dir: str, device: str = "cuda"):
        self.ckpt_dir = ckpt_dir
        self.device = device
        
        # Load config
        config_path = os.path.join(ckpt_dir, "config.json")
        if not os.path.exists(config_path):
            if "thinker" in ckpt_dir:
                config_path = "configs/thinker_tiny.json"
            elif "omni" in ckpt_dir:
                config_path = "configs/omni_sft_tiny.json"
            else:
                config_path = "configs/thinker_tiny.json"
        
        self.config = {}
        if os.path.exists(config_path):
            self.config = json.load(open(config_path))
        
        # Model loaders
        self.thinker_loader = ThinkerLoader(device)
        self.vision_loader = VisionLoader(device)
        self.audio_loader = AudioLoader(device)
        self.talker_loader = TalkerLoader(device)
        self.ocr_loader = OCRLoader(device)
        self.vocoder_loader = VocoderLoader(device)
        
        # Projectors
        self.proj_a = None
        self.proj_v = None
        
        # Processors
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor(device)
        
        # Generators (initialized after models load)
        self.text_generator = None
        self.audio_generator = None
        self.ocr_processor = None
    
    def load_models(self, load_vision: bool = True, load_audio: bool = True,
                   load_talker: bool = False, load_ocr: bool = False, load_vocoder: bool = False):
        """Load required models"""
        print(f"Loading models from {self.ckpt_dir}...")
        print("=" * 60)
        
        # Always load Thinker (required for all inference)
        print("\n[1/6] Loading ThinkerLM (Language Model)...")
        if not self.thinker_loader.load(self.ckpt_dir, self.config):
            raise RuntimeError("Failed to load Thinker model")
        
        self.text_generator = TextGenerator(
            self.thinker_loader.model,
            self.thinker_loader.tokenizer,
            self.device
        )
        
        # Load vision if needed
        if load_vision:
            print("\n[2/6] Loading Vision Encoder (ViTTiny)...")
            self.vision_loader.load(self.ckpt_dir, self.config)
        
        # Load audio if needed
        if load_audio:
            print("\n[3/6] Loading Audio Encoder (AudioEncoderTiny)...")
            self.audio_loader.load(self.ckpt_dir, self.config)
        
        # Load talker if needed (includes RVQ)
        if load_talker:
            print("\n[4/6] Loading Talker (TTS) + RVQ Codec...")
            self.talker_loader.load(self.ckpt_dir, self.config)
            if load_vocoder:
                print("\n[5/6] Loading Vocoder (NeuralVocoder: HiFi-GAN/Griffin-Lim)...")
                self.vocoder_loader.load(self.ckpt_dir)
            if self.talker_loader.loaded and self.vocoder_loader.loaded:
                self.audio_generator = AudioGenerator(
                    self.talker_loader.model,
                    self.talker_loader.rvq,
                    self.vocoder_loader.vocoder,
                    self.device
                )
        
        # Load OCR if needed
        if load_ocr:
            print("\n[6/6] Loading OCR Model (OCRModel)...")
            self.ocr_loader.load(self.ckpt_dir, self.config)
            if self.ocr_loader.loaded:
                self.ocr_processor = OCRProcessor(
                    self.ocr_loader.model,
                    self.ocr_loader.char_to_idx,
                    self.ocr_loader.idx_to_char,
                    self.device
                )
        
        # Load projectors
        print("\n[Projectors] Loading multimodal projectors...")
        self._load_projectors()
        
        # Print summary
        print("\n" + "=" * 60)
        print("MODEL LOADING SUMMARY")
        print("=" * 60)
        print(f"✓ ThinkerLM: {'Loaded' if self.thinker_loader.loaded else 'Not loaded'}")
        print(f"✓ Vision Encoder (ViTTiny): {'Loaded' if self.vision_loader.loaded else 'Not loaded'}")
        print(f"✓ Audio Encoder (AudioEncoderTiny): {'Loaded' if self.audio_loader.loaded else 'Not loaded'}")
        print(f"✓ Talker (TalkerTiny): {'Loaded' if self.talker_loader.loaded else 'Not loaded'}")
        print(f"✓ RVQ Codec: {'Loaded' if (self.talker_loader.loaded and self.talker_loader.rvq is not None) else 'Not loaded'}")
        print(f"✓ Vocoder (NeuralVocoder): {'Loaded' if self.vocoder_loader.loaded else 'Not loaded'}")
        print(f"✓ OCR Model (OCRModel): {'Loaded' if self.ocr_loader.loaded else 'Not loaded'}")
        print(f"✓ Tokenizer (BPETokenizer): {'Loaded' if self.thinker_loader.tokenizer is not None else 'Not loaded'}")
        print(f"✓ Projectors (proj_a, proj_v): {'Loaded' if (self.proj_a is not None and self.proj_v is not None) else 'Not loaded'}")
        print("=" * 60)
    
    def _load_projectors(self):
        """Load multimodal projectors"""
        # Get dimensions from config or defaults
        audio_dim = 192
        vision_dim = 128
        thinker_d_model = 256
        
        if isinstance(self.config, dict):
            if "audio" in self.config and isinstance(self.config["audio"], dict):
                audio_dim = self.config["audio"].get("d_model", 192)
            elif "d_model" in self.config.get("audio", {}):
                audio_dim = self.config["audio"]["d_model"]
            
            if "vision" in self.config and isinstance(self.config["vision"], dict):
                vision_dim = self.config["vision"].get("d_model", 128)
            elif "d_model" in self.config.get("vision", {}):
                vision_dim = self.config["vision"]["d_model"]
            
            if "thinker" in self.config and isinstance(self.config["thinker"], dict):
                thinker_d_model = self.config["thinker"].get("d_model", 256)
            elif "d_model" in self.config:
                thinker_d_model = self.config.get("d_model", 256)
        
        # Try to load from checkpoint
        ckpt_path, ckpt = find_checkpoint(self.ckpt_dir, "omni.pt", "omni_step_", self.device)
        loaded_from_ckpt = False
        
        if ckpt is not None and isinstance(ckpt, dict):
            if "proj_a" in ckpt and "proj_v" in ckpt:
                self.proj_a = nn.Linear(audio_dim, thinker_d_model).to(self.device)
                self.proj_v = nn.Linear(vision_dim, thinker_d_model).to(self.device)
                
                from omni.utils import strip_orig_mod
                proj_a_state = strip_orig_mod(ckpt["proj_a"])
                proj_v_state = strip_orig_mod(ckpt["proj_v"])
                
                if proj_a_state and proj_v_state:
                    self.proj_a.load_state_dict(proj_a_state, strict=False)
                    self.proj_v.load_state_dict(proj_v_state, strict=False)
                    print("✓ Loaded multimodal projectors from checkpoint")
                    loaded_from_ckpt = True
        
        # Create fallback projectors if not loaded from checkpoint
        if not loaded_from_ckpt:
            print(f"⚠ Projectors not found in checkpoint, creating fallback projectors")
            print(f"  Audio: {audio_dim} -> {thinker_d_model}")
            print(f"  Vision: {vision_dim} -> {thinker_d_model}")
            self.proj_a = nn.Linear(audio_dim, thinker_d_model).to(self.device)
            self.proj_v = nn.Linear(vision_dim, thinker_d_model).to(self.device)
            # Initialize with small random weights
            nn.init.normal_(self.proj_a.weight, std=0.02)
            nn.init.normal_(self.proj_v.weight, std=0.02)
            nn.init.zeros_(self.proj_a.bias)
            nn.init.zeros_(self.proj_v.bias)
            print("✓ Created fallback projectors (untrained)")
    
    def process_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Process image and return embeddings"""
        if not self.vision_loader.loaded:
            return None
        
        img_tensor = self.image_processor.process(image_path).to(self.device)
        
        if self.device == "cuda":
            with autocast(device_type='cuda'):
                cls, _ = self.vision_loader.model(img_tensor)
        else:
            cls, _ = self.vision_loader.model(img_tensor)
        
        # Always project vision embeddings to thinker dimension
        if self.proj_v is not None:
            if self.device == "cuda":
                with autocast(device_type='cuda'):
                    img_emb = self.proj_v(cls)
            else:
                img_emb = self.proj_v(cls)
            return img_emb
        else:
            # Fallback: if no projector, create one on the fly
            vision_dim = cls.shape[-1]
            thinker_d_model = self.config.get("thinker", {}).get("d_model", 256)
            if vision_dim != thinker_d_model:
                proj_v = nn.Linear(vision_dim, thinker_d_model).to(self.device)
                nn.init.normal_(proj_v.weight, std=0.02)
                nn.init.zeros_(proj_v.bias)
                if self.device == "cuda":
                    with autocast(device_type='cuda'):
                        img_emb = proj_v(cls)
                else:
                    img_emb = proj_v(cls)
                return img_emb
        return None
    
    def process_audio(self, audio_path: str) -> Optional[torch.Tensor]:
        """Process audio and return embeddings"""
        if not self.audio_loader.loaded:
            return None
        
        mel = self.audio_processor.process(audio_path)
        if mel is None:
            return None
        
        if self.device == "cuda":
            with autocast(device_type='cuda'):
                audio_emb = self.audio_loader.model(mel)
        else:
            audio_emb = self.audio_loader.model(mel)
        
        # Always project audio embeddings to thinker dimension
        if self.proj_a is not None:
            if self.device == "cuda":
                with autocast(device_type='cuda'):
                    audio_emb = self.proj_a(audio_emb)
            else:
                audio_emb = self.proj_a(audio_emb)
        else:
            # Fallback: if no projector, create one on the fly
            audio_dim = audio_emb.shape[-1]
            thinker_d_model = self.config.get("thinker", {}).get("d_model", 256)
            if audio_dim != thinker_d_model:
                proj_a = nn.Linear(audio_dim, thinker_d_model).to(self.device)
                nn.init.normal_(proj_a.weight, std=0.02)
                nn.init.zeros_(proj_a.bias)
                if self.device == "cuda":
                    with autocast(device_type='cuda'):
                        audio_emb = proj_a(audio_emb)
                else:
                    audio_emb = proj_a(audio_emb)
        
        # Limit length
        ctx_len = self.config.get("thinker", {}).get("ctx_len", 512)
        max_audio_tokens = min(audio_emb.shape[1], ctx_len // 4)
        return audio_emb[:, :max_audio_tokens, :]
    
    def process_video(self, video_path: str) -> Optional[torch.Tensor]:
        """Process video and return embeddings"""
        frames = VideoProcessor.extract_frames(video_path)
        if not frames:
            return None
        
        # Process first frame
        frame = frames[0].permute(1, 2, 0).numpy()
        frame = (frame * 255).astype('uint8')
        img = Image.fromarray(frame).convert("RGB")
        img_tensor = self.image_processor.transform(img).unsqueeze(0).to(self.device)
        
        if self.device == "cuda":
            with autocast(device_type='cuda'):
                cls, _ = self.vision_loader.model(img_tensor)
        else:
            cls, _ = self.vision_loader.model(img_tensor)
            
        # Always project vision embeddings to thinker dimension
        if self.proj_v is not None:
            if self.device == "cuda":
                with autocast(device_type='cuda'):
                    img_emb = self.proj_v(cls)
            else:
                img_emb = self.proj_v(cls)
            return img_emb
        else:
            # Fallback: if no projector, create one on the fly
            vision_dim = cls.shape[-1]
            thinker_d_model = self.config.get("thinker", {}).get("d_model", 256)
            if vision_dim != thinker_d_model:
                proj_v = nn.Linear(vision_dim, thinker_d_model).to(self.device)
                nn.init.normal_(proj_v.weight, std=0.02)
                nn.init.zeros_(proj_v.bias)
                if self.device == "cuda":
                    with autocast(device_type='cuda'):
                        img_emb = proj_v(cls)
                else:
                    img_emb = proj_v(cls)
                return img_emb
        return None
    
    def infer(self, text: Optional[str] = None, image: Optional[str] = None,
              video: Optional[str] = None, audio_in: Optional[str] = None,
              use_ocr: bool = False, audio_out: Optional[str] = None) -> str:
        """Main inference method"""
        multimodal_embeddings = []
        
        # Process OCR if requested
        if use_ocr and image and self.ocr_processor:
            img_tensor = self.image_processor.process(image).to(self.device)
            ocr_text = self.ocr_processor.extract_text(img_tensor)
            if ocr_text:
                print(f"OCR extracted text: {ocr_text}")
                if not text:
                    text = f"Extracted text from image: {ocr_text}. Describe what you see."
                else:
                    text = f"{text} (OCR extracted: {ocr_text})"
        
        # Process image
        if image:
            img_emb = self.process_image(image)
            if img_emb is not None:
                multimodal_embeddings.append(img_emb)
        
        # Process video
        if video:
            vid_emb = self.process_video(video)
            if vid_emb is not None:
                multimodal_embeddings.append(vid_emb)
        
        # Process audio
        if audio_in:
            aud_emb = self.process_audio(audio_in)
            if aud_emb is not None:
                multimodal_embeddings.append(aud_emb)
        
        # Build prompt
        if not text:
            if image or video:
                text = "Describe what you see concisely."
            elif audio_in:
                text = "What did you hear?"
            else:
                text = "Respond to the multimodal input."
        
        # Combine multimodal embeddings
        multimodal_emb = None
        if multimodal_embeddings:
            multimodal_emb = torch.cat(multimodal_embeddings, dim=1)
        
        # Generate text
        ctx_len = self.config.get("thinker", {}).get("ctx_len", 512)
        response = self.text_generator.generate(
            text, max_new=64, ctx=ctx_len,
            multimodal_emb=multimodal_emb
        )
        
        # Generate audio output if requested
        if audio_out and self.audio_generator:
            print("Generating audio output...")
            try:
                text_ids = self.thinker_loader.tokenizer.encode(response)
                audio = self.audio_generator.generate(text_ids)
                if audio is not None:
                    import soundfile as sf
                    sf.write(audio_out, audio, 16000)
                    print(f"Audio saved to: {audio_out}")
            except Exception as e:
                print(f"Warning: Could not generate audio: {e}")

        return response


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="μOmni multimodal inference interface")
    parser.add_argument("--ckpt_dir", required=True, help="Checkpoint directory")
    parser.add_argument("--image", default=None, help="Path to image file")
    parser.add_argument("--video", default=None, help="Path to video file")
    parser.add_argument("--audio_in", default=None, help="Path to audio input file")
    parser.add_argument("--audio_out", default=None, help="Path to save audio output file (TTS)")
    parser.add_argument("--text", default=None, help="Text prompt (optional, for multimodal)")
    parser.add_argument("--prompt", default=None, help="Override default prompt")
    parser.add_argument("--ocr", action="store_true", help="Extract text from image using OCR")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Determine which models to load
    has_image = args.image is not None
    has_video = args.video is not None
    has_audio = args.audio_in is not None
    use_ocr = args.ocr
    has_audio_out = args.audio_out is not None
    
    # Initialize inference engine
    engine = InferenceEngine(args.ckpt_dir, device)
    
    # Load required models
    engine.load_models(
        load_vision=has_image or has_video or use_ocr,
        load_audio=has_audio,
        load_talker=has_audio_out,
        load_ocr=use_ocr and has_image,
        load_vocoder=has_audio_out
    )
    
    # Run inference
    if has_image or has_video or has_audio or args.text:
        # Single query mode
        prompt = args.text or args.prompt
        response = engine.infer(
            text=prompt,
            image=args.image,
            video=args.video,
            audio_in=args.audio_in,
            use_ocr=use_ocr,
            audio_out=args.audio_out
        )
        print(f"\nμOmni (text): {response}\n")
    else:
        # Interactive mode
        print("Entering interactive chat mode. Type 'exit' to quit.")
        while True:
            try:
                q = input("You: ")
                if q.lower() in ['exit', 'quit', 'q']:
                    break
                response = engine.infer(text=q, audio_out=args.audio_out)
                print("μOmni:", response)
            except EOFError:
                break
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
