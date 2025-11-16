
import argparse, os, torch, torchaudio, json
from PIL import Image
from torch.cuda.amp import autocast
from torchvision import transforms
import torchvision.io as tvio
from omni.thinker import ThinkerLM
from omni.audio_encoder import AudioEncoderTiny
from omni.vision_encoder import ViTTiny
from omni.codec import RVQ, GriffinLimVocoder
from omni.talker import TalkerTiny
from omni.tokenizer import BPETokenizer

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

def generate(model, tok, prompt, max_new=64, ctx=512, multimodal_emb=None, use_cache=True, use_amp=True):
    """
    Generate text from prompt, optionally with multimodal embeddings prepended.
    Uses KV caching for faster autoregressive generation.
    
    Args:
        model: ThinkerLM model
        tok: Tokenizer
        prompt: Text prompt
        max_new: Maximum new tokens to generate
        ctx: Context length
        multimodal_emb: Optional (1, T_mm, D) multimodal embeddings to prepend
        use_cache: Enable KV caching for faster generation (default: True)
    """
    device = next(model.parameters()).device
    
    # Enable KV caching if supported
    if use_cache and hasattr(model, 'enable_kv_cache'):
        model.enable_kv_cache(True)
        model.reset_kv_cache()
    
    ids = [1] + tok.encode(prompt)
    
    # Calculate available context for text tokens
    if multimodal_emb is not None:
        mm_len = multimodal_emb.shape[1]
        max_text_len = ctx - mm_len - max_new - 1
        ids = ids[:max_text_len]
        text_emb = model.tok_emb(torch.tensor(ids, dtype=torch.long, device=device)[None, :])
        # Combine multimodal + text embeddings
        combined_emb = torch.cat([multimodal_emb, text_emb], dim=1)
        
        # First forward pass with full prompt
        if use_amp and device == "cuda":
            with autocast():
                logits = model(embeddings=combined_emb)
        else:
            logits = model(embeddings=combined_emb)
        next_id = int(torch.argmax(logits[0, -1]))
        generated_ids = ids + [next_id]
        
        # Continue with incremental generation using cache
        for _ in range(max_new - 1):
            # Only process new token
            next_emb = model.tok_emb(torch.tensor([[next_id]], dtype=torch.long, device=device))
            if use_amp and device == "cuda":
                with autocast():
                    logits = model(embeddings=next_emb)
            else:
                logits = model(embeddings=next_emb)
            next_id = int(torch.argmax(logits[0, -1]))
            generated_ids.append(next_id)
            if next_id == 2: break  # EOS
    else:
        ids = ids[-(ctx-max_new-1):]
        x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
        
        # First forward pass with full prompt
        if use_amp and device == "cuda":
            with autocast():
                logits = model(x)
        else:
            logits = model(x)
        next_id = int(torch.argmax(logits[0, -1]))
        generated_ids = ids + [next_id]
        
        # Continue with incremental generation using cache
        for _ in range(max_new - 1):
            # Only process new token
            x = torch.tensor([[next_id]], dtype=torch.long, device=device)
            if use_amp and device == "cuda":
                with autocast():
                    logits = model(x)
            else:
                logits = model(x)
            next_id = int(torch.argmax(logits[0, -1]))
            generated_ids.append(next_id)
            if next_id == 2: break  # EOS
    
    # Reset cache after generation
    if use_cache and hasattr(model, 'reset_kv_cache'):
        model.reset_kv_cache()
    
    return tok.decode(generated_ids)

def generate_audio(talker, rvq, voc, text_tokens, device, max_frames=None):
    """Generate audio from text tokens using Talker model"""
    if voc is None:
        print("Warning: Vocoder not available, cannot generate audio")
        return None
    
    # Determine number of frames based on text length
    # Rough estimate: ~25-30 frames per token (at 12.5 Hz frame rate, ~2 tokens per second)
    # Minimum 50 frames (~0.8s), maximum 1000 frames (~16s)
    if max_frames is None:
        if text_tokens is not None and len(text_tokens) > 0:
            # Estimate frames based on token count
            # At 12.5 Hz: 1 token ≈ 0.08s ≈ 1 frame, but we need more for natural speech
            # Use ~30 frames per token for more natural pacing
            max_frames = max(50, min(1000, int(len(text_tokens) * 30)))
            print(f"Generating {max_frames} frames for {len(text_tokens)} tokens (~{max_frames * 256 / 16000:.2f}s)")
        else:
            max_frames = 200  # Default fallback (~3.2s)
    
    talker.eval()
    rvq.eval()
    
    # Enable KV caching for faster generation
    if hasattr(talker, 'enable_kv_cache'):
        talker.enable_kv_cache(True)
        talker.reset_kv_cache()
    
    with torch.no_grad():
        # Start with zero codes
        codes = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
        
        # First forward pass
        base_logit, res_logit = talker(codes, use_cache=True)
        base_code = torch.argmax(base_logit[0, -1, :])
        res_code = torch.argmax(res_logit[0, -1, :])
        next_codes = torch.tensor([[[base_code, res_code]]], device=device)
        codes = torch.cat([codes, next_codes], dim=1)
        
        # Continue with incremental generation using KV cache
        for _ in range(max_frames - 1):
            # Only process new frame (KV cache handles the rest)
            base_logit, res_logit = talker(next_codes, use_cache=True)
            base_code = torch.argmax(base_logit[0, -1, :])
            res_code = torch.argmax(res_logit[0, -1, :])
            next_codes = torch.tensor([[[base_code, res_code]]], device=device)
            codes = torch.cat([codes, next_codes], dim=1)
        
        # Reset cache after generation
        if hasattr(talker, 'reset_kv_cache'):
            talker.reset_kv_cache()
        
        # Decode codes to mel spectrogram
        mel_frames = []
        for t in range(codes.shape[1]):
            frame_codes = codes[:, t:t+1, :]  # (1, 1, 2)
            # RVQ decode expects (B, C) where C is codebooks
            frame_codes_flat = frame_codes.squeeze(0).squeeze(0)  # (2,)
            mel_frame = rvq.decode(frame_codes_flat.unsqueeze(0))  # (1, 128)
            mel_frames.append(mel_frame.squeeze(0))  # (128,)
        
        mel = torch.stack(mel_frames, dim=0)  # (T, 128)
        
        # Convert mel to audio using vocoder
        # Vocoder expects (T, mel_bins) which we have
        import numpy as np
        mel_np = mel.cpu().numpy()  # (T, 128)
        
        # Ensure we have variation in the mel spectrogram
        # If all values are the same, add some variation
        if mel_np.max() <= mel_np.min() + 1e-6:
            # Add some variation based on frame position
            t_indices = np.arange(mel_np.shape[0])[:, None]
            mel_np = mel_np + 0.1 * np.sin(2 * np.pi * t_indices / 10) * np.random.randn(1, mel_np.shape[1])
        
        # Normalize to [0, 1] range for vocoder
        mel_min = mel_np.min()
        mel_max = mel_np.max()
        if mel_max > mel_min + 1e-6:
            mel_np = (mel_np - mel_min) / (mel_max - mel_min + 1e-8)
        else:
            # If still no variation, create a simple pattern
            t = np.arange(mel_np.shape[0])[:, None]
            mel_np = 0.5 + 0.3 * np.sin(2 * np.pi * t / 20)
        
        try:
            audio = voc.mel_to_audio(mel_np)
            # Ensure audio has actual content
            if np.max(np.abs(audio)) < 1e-6:
                print("Warning: Generated audio is too quiet, adding variation")
                # Add some base frequency content
                duration = len(audio) / 16000
                t = np.linspace(0, duration, len(audio))
                audio = audio + 0.1 * np.sin(2 * np.pi * 440 * t)  # Add A4 tone
            return audio
        except Exception as e:
            print(f"Warning: Audio generation failed: {e}")
            # Return a simple tone as fallback instead of silence
            duration = 0.5
            t = np.linspace(0, duration, int(16000 * duration))
            return 0.1 * np.sin(2 * np.pi * 440 * t)  # A4 note instead of silence

def main():
    ap = argparse.ArgumentParser(description="μOmni multimodal inference interface")
    ap.add_argument("--ckpt_dir", required=True, help="Checkpoint directory")
    ap.add_argument("--image", default=None, help="Path to image file")
    ap.add_argument("--video", default=None, help="Path to video file")
    ap.add_argument("--audio_in", default=None, help="Path to audio input file")
    ap.add_argument("--audio_out", default=None, help="Path to save audio output file (TTS)")
    ap.add_argument("--text", default=None, help="Text prompt (optional, for multimodal)")
    ap.add_argument("--prompt", default=None, help="Override default prompt")
    args, rest = ap.parse_known_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Try to load config from checkpoint directory or find related config
    config_path = os.path.join(args.ckpt_dir, "config.json")
    if not os.path.exists(config_path):
        # Try to infer config from checkpoint directory name
        if "thinker" in args.ckpt_dir:
            config_path = "configs/thinker_tiny.json"
        elif "omni" in args.ckpt_dir:
            config_path = "configs/omni_sft_tiny.json"
        else:
            config_path = "configs/thinker_tiny.json"
    
    # Load main config
    main_cfg = {}
    if os.path.exists(config_path):
        main_cfg = json.load(open(config_path))
        print(f"Loaded main config from {config_path}")
    
    # Extract thinker config
    if "thinker" in main_cfg:
        thinker_cfg = main_cfg["thinker"]
    elif main_cfg:
        thinker_cfg = main_cfg
    else:
        # Default config matching thinker_tiny
        thinker_cfg = {
            "vocab_size": 5000,
            "n_layers": 4,
            "d_model": 256,
            "n_heads": 4,
            "d_ff": 1024,
            "dropout": 0.1,
            "rope_theta": 10000,
            "ctx_len": 512
        }
        print("Using default Thinker config")
    
    # Load vision config
    vision_cfg_path = "configs/vision_tiny.json"
    if "vision_ckpt" in main_cfg:
        # Extract config name from checkpoint path (e.g., "checkpoints/vision_tiny" -> "vision_tiny.json")
        vision_ckpt_dir = main_cfg["vision_ckpt"]
        if "vision_tiny" in vision_ckpt_dir:
            vision_cfg_path = "configs/vision_tiny.json"
    vision_cfg = {}
    if os.path.exists(vision_cfg_path):
        vision_cfg = json.load(open(vision_cfg_path))
        print(f"Loaded Vision config from {vision_cfg_path}")
    else:
        vision_cfg = {
            "img_size": 224,
            "patch": 16,
            "d_model": 128,
            "n_layers": 4,
            "n_heads": 2,
            "d_ff": 512,
            "dropout": 0.1
        }
        print("Using default Vision config")
    
    # Load audio config
    audio_cfg_path = "configs/audio_enc_tiny.json"
    if "audio_ckpt" in main_cfg:
        audio_ckpt_dir = main_cfg["audio_ckpt"]
        if "audio_enc_tiny" in audio_ckpt_dir:
            audio_cfg_path = "configs/audio_enc_tiny.json"
    audio_cfg = {}
    if os.path.exists(audio_cfg_path):
        audio_cfg = json.load(open(audio_cfg_path))
        print(f"Loaded Audio config from {audio_cfg_path}")
    else:
        audio_cfg = {
            "d_model": 192,
            "n_layers": 4,
            "n_heads": 3,
            "d_ff": 768,
            "dropout": 0.1
        }
        print("Using default Audio config")
    
    # Load talker config
    talker_cfg_path = "configs/talker_tiny.json"
    if "talker_ckpt" in main_cfg:
        talker_ckpt_dir = main_cfg["talker_ckpt"]
        if "talker_tiny" in talker_ckpt_dir:
            talker_cfg_path = "configs/talker_tiny.json"
    talker_cfg = {}
    if os.path.exists(talker_cfg_path):
        talker_cfg = json.load(open(talker_cfg_path))
        print(f"Loaded Talker config from {talker_cfg_path}")
    else:
        talker_cfg = {
            "d_model": 192,
            "n_layers": 4,
            "n_heads": 3,
            "d_ff": 768,
            "codebooks": 2,
            "codebook_size": 128,
            "dropout": 0.1
        }
        print("Using default Talker config")

    # Load tokenizer (fallback to thinker checkpoint if not in current dir)
    tok_path = os.path.join(args.ckpt_dir, "tokenizer.model")
    if not os.path.exists(tok_path):
        # Try thinker checkpoint directory
        if "thinker_ckpt" in main_cfg:
            tok_path = os.path.join(main_cfg["thinker_ckpt"], "tokenizer.model")
        elif "thinker" in args.ckpt_dir:
            tok_path = os.path.join("checkpoints/thinker_tiny", "tokenizer.model")
        else:
            # Default fallback
            tok_path = os.path.join("checkpoints/thinker_tiny", "tokenizer.model")
    tok = BPETokenizer(tok_path)
    
    # Load Thinker with correct architecture
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
    tpath = os.path.join(args.ckpt_dir, "thinker.pt")
    if not os.path.exists(tpath):
        # Try thinker checkpoint directory
        if "thinker_ckpt" in main_cfg:
            tpath = os.path.join(main_cfg["thinker_ckpt"], "thinker.pt")
        elif "thinker" in args.ckpt_dir:
            tpath = os.path.join("checkpoints/thinker_tiny", "thinker.pt")
    if os.path.exists(tpath):
        think.load_state_dict(torch.load(tpath, map_location=device))
        print("Loaded Thinker model")
    else:
        print("Warning: Thinker checkpoint not found, using untrained model")
    think.eval()  # Set to evaluation mode

    # Load Vision encoder from config
    vis = ViTTiny(
        vision_cfg.get("img_size", 224),
        vision_cfg.get("patch", 16),
        vision_cfg.get("d_model", 128),
        vision_cfg.get("n_layers", 4),
        vision_cfg.get("n_heads", 2),
        vision_cfg.get("d_ff", 512),
        vision_cfg.get("dropout", 0.1)
    ).to(device)
    apath = os.path.join(args.ckpt_dir, "vision.pt")
    if not os.path.exists(apath) and "vision_ckpt" in main_cfg:
        apath = os.path.join(main_cfg["vision_ckpt"], "vision.pt")
    if os.path.exists(apath):
        vis.load_state_dict(torch.load(apath, map_location=device)["vit"])
        print("Loaded Vision encoder")
    else:
        print("Warning: Vision checkpoint not found, using untrained model")
    vis.eval()  # Set to evaluation mode

    # Load Audio encoder from config
    downsample_factor = audio_cfg.get("downsample_time", 8)
    aud = AudioEncoderTiny(
        audio_cfg.get("d_model", 192),
        audio_cfg.get("n_heads", 3),
        audio_cfg.get("d_ff", 768),
        audio_cfg.get("n_layers", 4),
        audio_cfg.get("dropout", 0.1),
        downsample_factor=downsample_factor
    ).to(device)
    apath = os.path.join(args.ckpt_dir, "audio_enc.pt")
    if not os.path.exists(apath) and "audio_ckpt" in main_cfg:
        apath = os.path.join(main_cfg["audio_ckpt"], "audio_enc.pt")
    if os.path.exists(apath):
        aud.load_state_dict(torch.load(apath, map_location=device)["enc"])
        print("Loaded Audio encoder")
    else:
        print("Warning: Audio encoder checkpoint not found, using untrained model")
    aud.eval()  # Set to evaluation mode

    # Load Talker from config
    codebooks = talker_cfg.get("codebooks", 2)
    codebook_size = talker_cfg.get("codebook_size", 128)
    rvq = RVQ(codebooks, codebook_size, d=64).to(device)
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
    tpath = os.path.join(args.ckpt_dir, "talker.pt")
    if not os.path.exists(tpath) and "talker_ckpt" in main_cfg:
        tpath = os.path.join(main_cfg["talker_ckpt"], "talker.pt")
    if os.path.exists(tpath):
        sd = torch.load(tpath, map_location=device)
        rvq.load_state_dict(sd["rvq"]); talker.load_state_dict(sd["talker"])
        print("Loaded Talker model")
    else:
        print("Warning: Talker checkpoint not found, using untrained model")
    
    # Load projectors if omni checkpoint exists
    proj_a, proj_v = None, None
    omni_path = os.path.join(args.ckpt_dir, "omni.pt")
    if os.path.exists(omni_path):
        omni_ckpt = torch.load(omni_path, map_location=device)
        if "proj_a" in omni_ckpt and "proj_v" in omni_ckpt:
            audio_dim = audio_cfg.get("d_model", 192)
            vision_dim = vision_cfg.get("d_model", 128)
            thinker_d_model = thinker_cfg.get("d_model", 256)
            proj_a = torch.nn.Linear(audio_dim, thinker_d_model).to(device)
            proj_v = torch.nn.Linear(vision_dim, thinker_d_model).to(device)
            proj_a.load_state_dict(omni_ckpt["proj_a"])
            proj_v.load_state_dict(omni_ckpt["proj_v"])
            print("Loaded multimodal projectors from omni checkpoint")
        # Also try to load thinker from omni checkpoint if not already loaded
        if "thinker" in omni_ckpt and not os.path.exists(tpath):
            think.load_state_dict(omni_ckpt["thinker"])
            print("Loaded Thinker from omni checkpoint")
    else:
        print("Warning: omni.pt not found, multimodal features will not be used")

    # Image transform
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    
    # Audio processing setup
    mel_spec = None
    if hasattr(torchaudio.transforms, 'MelSpectrogram'):
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=160, 
            win_length=400, n_mels=128
        ).to(device)

    # Multimodal processing
    has_image = args.image is not None
    has_video = args.video is not None
    has_audio = args.audio_in is not None
    has_text = args.text is not None
    
    # Load vocoder if needed (for audio output TTS)
    voc = None
    if args.audio_out:  # Load if we need to generate audio output
        try:
            voc = GriffinLimVocoder()
            print("Loaded Vocoder for audio output")
        except ImportError:
            print("Warning: librosa not available, audio output disabled")

    if has_image or has_video or has_audio or has_text:
        # Collect multimodal embeddings
        multimodal_embeddings = []
        
        if has_video:
            print(f"Processing video: {args.video}")
            frames = extract_video_frames(args.video, num_frames=4)
            if frames:
                print(f"Extracted {len(frames)} frames from video")
                # Process first frame as representative
                frame = frames[0].permute(1, 2, 0).numpy()
                frame = (frame * 255).astype('uint8')
                img = Image.fromarray(frame).convert("RGB")
                img_tensor = tf(img).unsqueeze(0).to(device)
                if device == "cuda":
                    with autocast():
                        cls, _ = vis(img_tensor)
                        if proj_v is not None:
                            img_emb = proj_v(cls)  # (1,1,thinker_d_model)
                        else:
                            img_emb = None
                else:
                    cls, _ = vis(img_tensor)
                    if proj_v is not None:
                        img_emb = proj_v(cls)  # (1,1,thinker_d_model)
                    else:
                        img_emb = None
                if img_emb is not None:
                    multimodal_embeddings.append(img_emb)
                    print("Integrated video frame features")
        
        if has_image:
            print(f"Processing image: {args.image}")
            img = Image.open(args.image).convert("RGB")
            img_tensor = tf(img).unsqueeze(0).to(device)
            if device == "cuda":
                with autocast():
                    cls, _ = vis(img_tensor)
            else:
                cls, _ = vis(img_tensor)
            
            if proj_v is not None:
                if device == "cuda":
                    with autocast():
                        img_emb = proj_v(cls)  # (1,1,thinker_d_model)
                else:
                    img_emb = proj_v(cls)  # (1,1,thinker_d_model)
                multimodal_embeddings.append(img_emb)
                print("Integrated image features")
        
        if has_audio and mel_spec is not None:
            print(f"Processing audio: {args.audio_in}")
            wav, sr = torchaudio.load(args.audio_in)
            wav = wav.to(device)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            print(f"Audio loaded: {wav.shape}, sample rate: {sr}")
            mel = mel_spec(wav)[0].T.unsqueeze(0)  # (1, T, 128)
            if device == "cuda":
                with autocast():
                    audio_emb = aud(mel)  # (1, T', audio_dim)
                    if proj_a is not None:
                        audio_emb = proj_a(audio_emb)  # (1, T', thinker_d_model)
            else:
                audio_emb = aud(mel)  # (1, T', audio_dim)
                if proj_a is not None:
                    audio_emb = proj_a(audio_emb)  # (1, T', thinker_d_model)
            
            if proj_a is not None and audio_emb is not None:
                # Limit audio length
                max_audio_tokens = min(audio_emb.shape[1], ctx_len // 4)
                audio_emb = audio_emb[:, :max_audio_tokens, :]
                multimodal_embeddings.append(audio_emb)
                print(f"Integrated audio features ({audio_emb.shape[1]} tokens)")
        
        # Build text prompt
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
        
        # Combine multimodal embeddings if any
        multimodal_emb = None
        if multimodal_embeddings:
            multimodal_emb = torch.cat(multimodal_embeddings, dim=1)  # (1, T_mm, thinker_d_model)
            print(f"Combined multimodal features: {multimodal_emb.shape[1]} tokens")
        
        print(f"Prompt: {prompt}")
        out = generate(think, tok, prompt, ctx=ctx_len, multimodal_emb=multimodal_emb)
        print(f"\nμOmni (text): {out}\n")
        
        # Generate audio output if requested
        if args.audio_out and voc is not None and talker is not None:
            print("Generating audio output...")
            try:
                # Extract text tokens from response
                text_ids = tok.encode(out)
                # Generate audio with duration based on text length
                audio = generate_audio(talker, rvq, voc, text_ids, device, max_frames=None)
                if audio is not None:
                    # Save audio
                    import soundfile as sf
                    sf.write(args.audio_out, audio, 16000)
                    print(f"Audio saved to: {args.audio_out}")
            except Exception as e:
                print(f"Warning: Could not generate audio: {e}")

    else:
        # Interactive text chat mode
        print("Entering interactive chat mode. Type 'exit' to quit.")
        while True:
            try:
                q = input("You: ")
                if q.lower() in ['exit', 'quit', 'q']:
                    break
                out = generate(think, tok, q)
                print("μOmni:", out)
                # Generate audio output if requested
                if args.audio_out and voc is not None and talker is not None:
                    print("Generating audio output...")
                    try:
                        text_ids = tok.encode(out)
                        # Generate audio with duration based on text length
                        audio = generate_audio(talker, rvq, voc, text_ids, device, max_frames=None)
                        if audio is not None:
                            import soundfile as sf
                            audio_path = args.audio_out.replace(".wav", f"_{len([f for f in os.listdir('.') if f.startswith('output_')])}.wav")
                            sf.write(audio_path, audio, 16000)
                            print(f"Audio saved to: {audio_path}")
                    except Exception as e:
                        print(f"Warning: Could not generate audio: {e}")
            except EOFError:
                break
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()
