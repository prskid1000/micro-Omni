"""
Inference script for μOmni model loaded from safetensors file.

This script loads all model components from a single safetensors file
and performs multimodal inference (text, image, audio, video).
"""

import argparse
import os
import torch
import torchaudio
import json
from PIL import Image
from torch.amp import autocast
from torchvision import transforms
import torchvision.io as tvio
from safetensors.torch import load_file

from omni.thinker import ThinkerLM
from omni.audio_encoder import AudioEncoderTiny
from omni.vision_encoder import ViTTiny
from omni.codec import RVQ, NeuralVocoder
from omni.talker import TalkerTiny
from omni.tokenizer import BPETokenizer
from omni.ocr_model import OCRModel


def extract_video_frames(video_path, num_frames=4):
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


def load_model_from_safetensors(model_dir, device="cuda"):
    """
    Load all model components from safetensors file and configs.
    
    Args:
        model_dir: Directory containing model.safetensors and support files
        device: Device to load models on
    
    Returns:
        Dictionary containing all loaded models and components
    """
    model_dir = os.path.abspath(model_dir)
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    
    if not os.path.exists(safetensors_path):
        # Try alternative names
        for alt_name in ["muomni.safetensors", "omni.safetensors"]:
            alt_path = os.path.join(model_dir, alt_name)
            if os.path.exists(alt_path):
                safetensors_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Could not find safetensors file in {model_dir}")
    
    print(f"Loading model from {safetensors_path}...")
    state_dict = load_file(safetensors_path, device=device)
    print(f"  Loaded {len(state_dict)} parameter tensors")
    
    # Load model info if available
    info_path = os.path.join(model_dir, "model_info.json")
    model_info = {}
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            model_info = json.load(f)
        print(f"  Model info: {model_info.get('model_type', 'unknown')}")
    
    # Load configs
    configs_dir = os.path.join(model_dir, "configs")
    if not os.path.exists(configs_dir):
        configs_dir = "configs"  # Fallback to default
    
    # Load Thinker config
    thinker_cfg_path = os.path.join(configs_dir, "thinker_tiny.json")
    if not os.path.exists(thinker_cfg_path):
        thinker_cfg_path = "configs/thinker_tiny.json"
    thinker_cfg = json.load(open(thinker_cfg_path)) if os.path.exists(thinker_cfg_path) else {}
    
    # Load Vision config
    vision_cfg_path = os.path.join(configs_dir, "vision_tiny.json")
    if not os.path.exists(vision_cfg_path):
        vision_cfg_path = "configs/vision_tiny.json"
    vision_cfg = json.load(open(vision_cfg_path)) if os.path.exists(vision_cfg_path) else {}
    
    # Load Audio config
    audio_cfg_path = os.path.join(configs_dir, "audio_enc_tiny.json")
    if not os.path.exists(audio_cfg_path):
        audio_cfg_path = "configs/audio_enc_tiny.json"
    audio_cfg = json.load(open(audio_cfg_path)) if os.path.exists(audio_cfg_path) else {}
    
    # Load Talker config
    talker_cfg_path = os.path.join(configs_dir, "talker_tiny.json")
    if not os.path.exists(talker_cfg_path):
        talker_cfg_path = "configs/talker_tiny.json"
    talker_cfg = json.load(open(talker_cfg_path)) if os.path.exists(talker_cfg_path) else {}
    
    # Extract component state dicts
    def extract_component(prefix):
        component_dict = {}
        for key, value in state_dict.items():
            if key.startswith(prefix + "."):
                new_key = key[len(prefix) + 1:]  # Remove prefix and dot
                component_dict[new_key] = value
        return component_dict
    
    models = {}
    
    # 1. Load Thinker
    print("  Loading Thinker...")
    thinker_state = extract_component("thinker")
    if not thinker_state:
        raise ValueError("Thinker weights not found in safetensors file")
    
    ctx_len = thinker_cfg.get("ctx_len", 512)
    think = ThinkerLM(
        thinker_cfg.get("vocab_size", 5000),
        thinker_cfg.get("n_layers", 4),
        thinker_cfg.get("d_model", 256),
        thinker_cfg.get("n_heads", 4),
        thinker_cfg.get("d_ff", 1024),
        thinker_cfg.get("dropout", 0.1),
        thinker_cfg.get("rope_theta", 10000),
        ctx_len,
        use_gqa=thinker_cfg.get("use_gqa", False),
        use_swiglu=thinker_cfg.get("use_swiglu", True),
        use_moe=thinker_cfg.get("use_moe", False),
        num_experts=thinker_cfg.get("num_experts", 8),
        num_experts_per_tok=thinker_cfg.get("num_experts_per_tok", 2)
    ).to(device)
    think.load_state_dict(thinker_state)
    think.eval()
    models["thinker"] = think
    print("    ✓ Thinker loaded")
    
    # 2. Load Vision Encoder
    print("  Loading Vision Encoder...")
    vision_state = extract_component("vision_encoder")
    if vision_state:
        vis = ViTTiny(
            vision_cfg.get("img_size", 224),
            vision_cfg.get("patch", 16),
            vision_cfg.get("d_model", 128),
            vision_cfg.get("n_layers", 4),
            vision_cfg.get("n_heads", 2),
            vision_cfg.get("d_ff", 512),
            vision_cfg.get("dropout", 0.1)
        ).to(device)
        vis.load_state_dict(vision_state)
        vis.eval()
        models["vision"] = vis
        print("    ✓ Vision Encoder loaded")
    else:
        print("    ⚠ Vision Encoder not found")
        models["vision"] = None
    
    # 3. Load Audio Encoder
    print("  Loading Audio Encoder...")
    audio_state = extract_component("audio_encoder")
    if audio_state:
        downsample_factor = audio_cfg.get("downsample_time", 8)
        aud = AudioEncoderTiny(
            audio_cfg.get("d_model", 192),
            audio_cfg.get("n_heads", 3),
            audio_cfg.get("d_ff", 768),
            audio_cfg.get("n_layers", 4),
            audio_cfg.get("dropout", 0.1),
            downsample_factor=downsample_factor
        ).to(device)
        aud.load_state_dict(audio_state)
        aud.eval()
        models["audio"] = aud
        print("    ✓ Audio Encoder loaded")
    else:
        print("    ⚠ Audio Encoder not found")
        models["audio"] = None
    
    # 4. Load Talker + RVQ
    print("  Loading Talker...")
    talker_state = extract_component("talker")
    rvq_state = extract_component("rvq")
    if talker_state and rvq_state:
        codebooks = talker_cfg.get("codebooks", 2)
        codebook_size = talker_cfg.get("codebook_size", 128)
        rvq = RVQ(codebooks, codebook_size, d=64).to(device)
        rvq.load_state_dict(rvq_state)
        
        talker = TalkerTiny(
            talker_cfg.get("d_model", 192),
            talker_cfg.get("n_layers", 4),
            talker_cfg.get("n_heads", 3),
            talker_cfg.get("d_ff", 768),
            codebooks,
            codebook_size,
            talker_cfg.get("dropout", 0.1),
            use_gqa=talker_cfg.get("use_gqa", False),
            use_swiglu=talker_cfg.get("use_swiglu", True),
            rope_theta=talker_cfg.get("rope_theta", 10000.0)
        ).to(device)
        talker.load_state_dict(talker_state)
        talker.eval()
        rvq.eval()
        models["talker"] = talker
        models["rvq"] = rvq
        print("    ✓ Talker + RVQ loaded")
    else:
        print("    ⚠ Talker not found")
        models["talker"] = None
        models["rvq"] = None
    
    # 5. Load Projectors
    print("  Loading Projectors...")
    proj_a_state = extract_component("proj_a")
    proj_v_state = extract_component("proj_v")
    
    proj_a, proj_v = None, None
    if proj_a_state and proj_v_state:
        audio_dim = audio_cfg.get("d_model", 192)
        vision_dim = vision_cfg.get("d_model", 128)
        thinker_d_model = thinker_cfg.get("d_model", 256)
        
        proj_a = torch.nn.Linear(audio_dim, thinker_d_model).to(device)
        proj_v = torch.nn.Linear(vision_dim, thinker_d_model).to(device)
        proj_a.load_state_dict(proj_a_state)
        proj_v.load_state_dict(proj_v_state)
        proj_a.eval()
        proj_v.eval()
        models["proj_a"] = proj_a
        models["proj_v"] = proj_v
        print("    ✓ Projectors loaded")
    else:
        print("    ⚠ Projectors not found")
        models["proj_a"] = None
        models["proj_v"] = None
    
    # 6. Load OCR (optional)
    print("  Loading OCR...")
    ocr_state = extract_component("ocr")
    if ocr_state:
        ocr_cfg_path = os.path.join(configs_dir, "ocr_tiny.json")
        if not os.path.exists(ocr_cfg_path):
            ocr_cfg_path = "configs/ocr_tiny.json"
        ocr_cfg = json.load(open(ocr_cfg_path)) if os.path.exists(ocr_cfg_path) else {}
        
        ocr_model = OCRModel(
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
            vocab_size=ocr_cfg.get("vocab_size", 128),
            dropout=ocr_cfg.get("dropout", 0.1)
        ).to(device)
        ocr_model.load_state_dict(ocr_state)
        ocr_model.eval()
        models["ocr"] = ocr_model
        print("    ✓ OCR loaded")
    else:
        print("    ⚠ OCR not found (optional)")
        models["ocr"] = None
    
    # Load tokenizer
    tokenizer_path = os.path.join(model_dir, "tokenizer.model")
    if not os.path.exists(tokenizer_path):
        tokenizer_path = os.path.join(model_dir, "..", "checkpoints", "thinker_tiny", "tokenizer.model")
    if os.path.exists(tokenizer_path):
        models["tokenizer"] = BPETokenizer(tokenizer_path)
        print("  ✓ Tokenizer loaded")
    else:
        raise FileNotFoundError(f"Tokenizer not found. Expected at {tokenizer_path}")
    
    # Load vocoder if available
    hifigan_path = os.path.join(model_dir, "hifigan.pt")
    if os.path.exists(hifigan_path):
        try:
            voc = NeuralVocoder(
                sample_rate=16000,
                n_mels=128,
                checkpoint_path=hifigan_path,
                prefer_neural=True
            )
            if voc.vocoder_type == "hifigan":
                voc = voc.to(device).eval()
            models["vocoder"] = voc
            print("  ✓ Vocoder loaded")
        except Exception as e:
            print(f"  ⚠ Warning: Could not load vocoder: {e}")
            models["vocoder"] = None
    else:
        models["vocoder"] = None
    
    print("\n✓ All models loaded successfully!")
    return models


def generate(model, tok, prompt, max_new=64, ctx=512, multimodal_emb=None, use_cache=True, use_amp=True):
    """Generate text from prompt, optionally with multimodal embeddings prepended."""
    device = next(model.parameters()).device
    
    if use_cache and hasattr(model, 'enable_kv_cache'):
        model.enable_kv_cache(True)
        model.reset_kv_cache()
    
    ids = [1] + tok.encode(prompt)
    
    if multimodal_emb is not None:
        mm_len = multimodal_emb.shape[1]
        max_text_len = ctx - mm_len - max_new - 1
        ids = ids[:max_text_len]
        text_emb = model.tok_emb(torch.tensor(ids, dtype=torch.long, device=device)[None, :])
        combined_emb = torch.cat([multimodal_emb, text_emb], dim=1)
        
        if use_amp and device == "cuda":
            with autocast(device_type='cuda'):
                logits = model(embeddings=combined_emb)
        else:
            logits = model(embeddings=combined_emb)
        next_id = int(torch.argmax(logits[0, -1]))
        generated_ids = ids + [next_id]
        
        for _ in range(max_new - 1):
            next_emb = model.tok_emb(torch.tensor([[next_id]], dtype=torch.long, device=device))
            if use_amp and device == "cuda":
                with autocast(device_type='cuda'):
                    logits = model(embeddings=next_emb)
            else:
                logits = model(embeddings=next_emb)
            next_id = int(torch.argmax(logits[0, -1]))
            generated_ids.append(next_id)
            if next_id == 2: break
    else:
        ids = ids[-(ctx-max_new-1):]
        x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
        
        if use_amp and device == "cuda":
            with autocast(device_type='cuda'):
                logits = model(x)
        else:
            logits = model(x)
        next_id = int(torch.argmax(logits[0, -1]))
        generated_ids = ids + [next_id]
        
        for _ in range(max_new - 1):
            x = torch.tensor([[next_id]], dtype=torch.long, device=device)
            if use_amp and device == "cuda":
                with autocast(device_type='cuda'):
                    logits = model(x)
            else:
                logits = model(x)
            next_id = int(torch.argmax(logits[0, -1]))
            generated_ids.append(next_id)
            if next_id == 2: break
    
    if use_cache and hasattr(model, 'reset_kv_cache'):
        model.reset_kv_cache()
    
    return tok.decode(generated_ids)


def generate_audio(talker, rvq, voc, text_tokens, device, max_frames=None):
    """Generate audio from text tokens using Talker model"""
    if voc is None:
        print("Warning: Vocoder not available, cannot generate audio")
        return None
    
    if max_frames is None:
        if text_tokens is not None and len(text_tokens) > 0:
            max_frames = max(50, min(1000, int(len(text_tokens) * 30)))
        else:
            max_frames = 200
    
    talker.eval()
    rvq.eval()
    
    if hasattr(talker, 'enable_kv_cache'):
        talker.enable_kv_cache(True)
        talker.reset_kv_cache()
    
    with torch.no_grad():
        codes = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
        
        base_logit, res_logit = talker(codes, use_cache=True)
        base_code = torch.argmax(base_logit[0, -1, :])
        res_code = torch.argmax(res_logit[0, -1, :])
        next_codes = torch.tensor([[[base_code, res_code]]], device=device)
        codes = torch.cat([codes, next_codes], dim=1)
        
        for _ in range(max_frames - 1):
            base_logit, res_logit = talker(next_codes, use_cache=True)
            base_code = torch.argmax(base_logit[0, -1, :])
            res_code = torch.argmax(res_logit[0, -1, :])
            next_codes = torch.tensor([[[base_code, res_code]]], device=device)
            codes = torch.cat([codes, next_codes], dim=1)
        
        if hasattr(talker, 'reset_kv_cache'):
            talker.reset_kv_cache()
        
        mel_frames = []
        for t in range(codes.shape[1]):
            frame_codes = codes[:, t:t+1, :]
            frame_codes_flat = frame_codes.squeeze(0).squeeze(0)
            mel_frame = rvq.decode(frame_codes_flat.unsqueeze(0))
            mel_frames.append(mel_frame.squeeze(0))
        
        mel = torch.stack(mel_frames, dim=0)
        
        import numpy as np
        mel_min = mel.min()
        mel_max = mel.max()
        if mel_max > mel_min + 1e-6:
            mel_normalized = (mel - mel_min) / (mel_max - mel_min + 1e-8)
        else:
            t = torch.arange(mel.shape[0], device=mel.device, dtype=mel.dtype)[:, None]
            mel_normalized = 0.5 + 0.3 * torch.sin(2 * np.pi * t / 20)
        
        try:
            if voc.vocoder_type == "hifigan" and isinstance(voc.vocoder, torch.nn.Module):
                mel_tensor = mel_normalized.T.unsqueeze(0)
                with torch.no_grad():
                    audio_tensor = voc.vocoder(mel_tensor)
                audio = audio_tensor.squeeze().cpu().numpy()
            else:
                mel_np = mel_normalized.cpu().numpy()
                audio = voc.mel_to_audio(mel_np)
            
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


def main():
    ap = argparse.ArgumentParser(description="μOmni inference from safetensors")
    ap.add_argument("--model_dir", required=True, help="Directory containing model.safetensors and support files")
    ap.add_argument("--image", default=None, help="Path to image file")
    ap.add_argument("--video", default=None, help="Path to video file")
    ap.add_argument("--audio_in", default=None, help="Path to audio input file")
    ap.add_argument("--audio_out", default=None, help="Path to save audio output file (TTS)")
    ap.add_argument("--text", default=None, help="Text prompt")
    ap.add_argument("--prompt", default=None, help="Override default prompt")
    ap.add_argument("--ocr", action="store_true", help="Extract text from image using OCR")
    args = ap.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load all models
    models = load_model_from_safetensors(args.model_dir, device)
    
    think = models["thinker"]
    vis = models["vision"]
    aud = models["audio"]
    talker = models["talker"]
    rvq = models["rvq"]
    proj_a = models["proj_a"]
    proj_v = models["proj_v"]
    tok = models["tokenizer"]
    voc = models["vocoder"]
    ocr_model = models.get("ocr", None)
    
    # Get context length from config
    configs_dir = os.path.join(args.model_dir, "configs")
    if not os.path.exists(configs_dir):
        configs_dir = "configs"
    thinker_cfg_path = os.path.join(configs_dir, "thinker_tiny.json")
    if not os.path.exists(thinker_cfg_path):
        thinker_cfg_path = "configs/thinker_tiny.json"
    thinker_cfg = json.load(open(thinker_cfg_path)) if os.path.exists(thinker_cfg_path) else {}
    ctx_len = thinker_cfg.get("ctx_len", 512)
    
    # Image transform
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    
    # Audio processing
    mel_spec = None
    if hasattr(torchaudio.transforms, 'MelSpectrogram'):
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=160,
            win_length=400, n_mels=128
        ).to(device)
    
    # Process inputs
    has_image = args.image is not None
    has_video = args.video is not None
    has_audio = args.audio_in is not None
    has_text = args.text is not None
    
    if has_image or has_video or has_audio or has_text:
        multimodal_embeddings = []
        
        if has_video and vis is not None:
            print(f"Processing video: {args.video}")
            frames = extract_video_frames(args.video, num_frames=4)
            if frames:
                frame = frames[0].permute(1, 2, 0).numpy()
                frame = (frame * 255).astype('uint8')
                img = Image.fromarray(frame).convert("RGB")
                img_tensor = tf(img).unsqueeze(0).to(device)
                if device == "cuda":
                    with autocast(device_type='cuda'):
                        cls, _ = vis(img_tensor)
                        if proj_v is not None:
                            img_emb = proj_v(cls)
                        else:
                            img_emb = None
                else:
                    cls, _ = vis(img_tensor)
                    if proj_v is not None:
                        img_emb = proj_v(cls)
                    else:
                        img_emb = None
                if img_emb is not None:
                    multimodal_embeddings.append(img_emb)
                    print("Integrated video frame features")
        
        if has_image and vis is not None:
            print(f"Processing image: {args.image}")
            img = Image.open(args.image).convert("RGB")
            img_tensor = tf(img).unsqueeze(0).to(device)
            
            if args.ocr and ocr_model is not None:
                print("Extracting text from image using OCR...")
                try:
                    ocr_model.eval()
                    max_ocr_len = 128
                    with torch.no_grad():
                        # Try to load OCR char mappings from checkpoint if available
                        ocr_char_to_idx = None
                        ocr_idx_to_char = None
                        ocr_ckpt_path = os.path.join(args.model_dir, "ocr.pt")
                        if os.path.exists(ocr_ckpt_path):
                            ocr_ckpt = torch.load(ocr_ckpt_path, map_location=device)
                            if "char_to_idx" in ocr_ckpt:
                                ocr_char_to_idx = ocr_ckpt["char_to_idx"]
                                ocr_idx_to_char = ocr_ckpt.get("idx_to_char", {v: k for k, v in ocr_char_to_idx.items()})
                        
                        bos_id = ocr_char_to_idx.get('<BOS>', 1) if ocr_char_to_idx else 1
                        eos_id = ocr_char_to_idx.get('<EOS>', 2) if ocr_char_to_idx else 2
                        
                        text_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)
                        ocr_text = ""
                        
                        for _ in range(max_ocr_len):
                            if device == "cuda":
                                with autocast(device_type='cuda'):
                                    logits = ocr_model(img_tensor, text_ids)
                            else:
                                logits = ocr_model(img_tensor, text_ids)
                            
                            next_id = int(torch.argmax(logits[0, -1]))
                            if next_id == eos_id or next_id == 0:
                                break
                            
                            if ocr_idx_to_char is not None:
                                char = ocr_idx_to_char.get(next_id, '')
                                if char and char not in ['<BOS>', '<EOS>', '<PAD>', '<UNK>']:
                                    ocr_text += char
                            else:
                                if 32 <= next_id < 127:
                                    ocr_text += chr(next_id)
                            
                            text_ids = torch.cat([text_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
                        
                        if ocr_text:
                            print(f"OCR extracted text: {ocr_text}")
                            if not has_text:
                                args.text = f"Extracted text from image: {ocr_text}. Describe what you see."
                            else:
                                args.text = f"{args.text} (OCR extracted: {ocr_text})"
                except Exception as e:
                    print(f"Warning: OCR extraction failed: {e}")
            
            if device == "cuda":
                with autocast(device_type='cuda'):
                    cls, _ = vis(img_tensor)
            else:
                cls, _ = vis(img_tensor)
            
            if proj_v is not None:
                if device == "cuda":
                    with autocast(device_type='cuda'):
                        img_emb = proj_v(cls)
                else:
                    img_emb = proj_v(cls)
                multimodal_embeddings.append(img_emb)
                print("Integrated image features")
        
        if has_audio and aud is not None and mel_spec is not None:
            print(f"Processing audio: {args.audio_in}")
            wav, sr = torchaudio.load(args.audio_in)
            wav = wav.to(device)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            mel = mel_spec(wav)[0].T.unsqueeze(0)
            if device == "cuda":
                with autocast(device_type='cuda'):
                    audio_emb = aud(mel)
                    if proj_a is not None:
                        audio_emb = proj_a(audio_emb)
            else:
                audio_emb = aud(mel)
                if proj_a is not None:
                    audio_emb = proj_a(audio_emb)
            
            if proj_a is not None and audio_emb is not None:
                max_audio_tokens = min(audio_emb.shape[1], ctx_len // 4)
                audio_emb = audio_emb[:, :max_audio_tokens, :]
                multimodal_embeddings.append(audio_emb)
                print(f"Integrated audio features ({audio_emb.shape[1]} tokens)")
        
        if has_text:
            prompt = args.text
        elif args.prompt:
            prompt = args.prompt
        elif has_image or has_video:
            prompt = "Describe what you see concisely."
        elif has_audio:
            prompt = "What did you hear?"
        else:
            prompt = "Respond to the multimodal input."
        
        multimodal_emb = None
        if multimodal_embeddings:
            multimodal_emb = torch.cat(multimodal_embeddings, dim=1)
            print(f"Combined multimodal features: {multimodal_emb.shape[1]} tokens")
        
        print(f"Prompt: {prompt}")
        out = generate(think, tok, prompt, ctx=ctx_len, multimodal_emb=multimodal_emb)
        print(f"\nμOmni (text): {out}\n")
        
        if args.audio_out and voc is not None and talker is not None:
            print("Generating audio output...")
            try:
                text_ids = tok.encode(out)
                audio = generate_audio(talker, rvq, voc, text_ids, device, max_frames=None)
                if audio is not None:
                    import soundfile as sf
                    sf.write(args.audio_out, audio, 16000)
                    print(f"Audio saved to: {args.audio_out}")
            except Exception as e:
                print(f"Warning: Could not generate audio: {e}")
    else:
        # Interactive chat mode
        print("Entering interactive chat mode. Type 'exit' to quit.")
        while True:
            try:
                q = input("You: ")
                if q.lower() in ['exit', 'quit', 'q']:
                    break
                out = generate(think, tok, q)
                print("μOmni:", out)
            except EOFError:
                break
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()

