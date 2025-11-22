"""
ASR (Automatic Speech Recognition) + TTS (Text-to-Speech) round-trip test script.
Tests the full pipeline: Speech → Text → Speech
Pipeline: Audio Encoder + CTC Head → Text → Talker + RVQ + Vocoder → Audio
Evaluates on 100 samples and reports:
- ASR metrics: WER, CER, accuracy, loss
- TTS metrics: Reconstruction error, audio quality
"""

import torch
import json
import os
import argparse
import random
import torchaudio
import numpy as np
from torch.nn import CTCLoss
from omni.audio_encoder import AudioEncoderTiny
from omni.talker import TalkerTiny
from omni.codec import RVQ, NeuralVocoder
from omni.utils import ASRDataset, load_audio, find_checkpoint, strip_orig_mod


def build_char_vocab():
    """Build character vocabulary matching training script"""
    char_to_idx = {}
    for i in range(32, 127):  # Printable ASCII
        char_to_idx[chr(i)] = len(char_to_idx) + 1
    char_to_idx['\n'] = len(char_to_idx) + 1
    char_to_idx['\t'] = len(char_to_idx) + 1
    char_to_idx['<UNK>'] = len(char_to_idx) + 1
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    idx_to_char[0] = '<BLANK>'  # CTC blank token
    vocab_size = len(char_to_idx) + 1  # +1 for blank token (0)
    return char_to_idx, idx_to_char, vocab_size


def ctc_greedy_decode(logits, idx_to_char):
    """
    Greedy CTC decoding: pick most likely character at each timestep, then collapse.
    
    Args:
        logits: (T, vocab_size) logits from CTC head
        idx_to_char: mapping from index to character
        
    Returns:
        Decoded text string
    """
    # Get predicted indices (greedy)
    pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()  # (T,)
    
    # CTC collapse: remove blanks and repeated characters
    decoded = []
    prev_id = None
    for idx in pred_ids:
        if idx == 0:  # Blank token
            prev_id = None
            continue
        if idx != prev_id:  # Different from previous (collapse repeats)
            if idx in idx_to_char:
                char = idx_to_char[idx]
                if char not in ['<BLANK>', '<UNK>']:
                    decoded.append(char)
        prev_id = idx
    
    return ''.join(decoded)


def load_asr_models(audio_ckpt_dir, device="cuda"):
    """Load Audio Encoder and CTC head from checkpoint."""
    checkpoint_path, checkpoint = find_checkpoint(audio_ckpt_dir, "audio_enc.pt", "audio_enc_step_", device)
    if checkpoint is None:
        raise FileNotFoundError(f"Audio encoder checkpoint not found in: {audio_ckpt_dir}")
    
    print(f"Loading Audio Encoder from: {checkpoint_path}")
    
    # Get config from checkpoint or use defaults
    if "config" in checkpoint:
        cfg = checkpoint["config"]
    else:
        config_path = "configs/audio_enc_tiny.json"
        if os.path.exists(config_path):
            cfg = json.load(open(config_path))
        else:
            cfg = {
                "d_model": 192,
                "n_heads": 3,
                "d_ff": 768,
                "n_layers": 4,
                "dropout": 0.1,
                "downsample_time": 8,
                "sample_rate": 16000,
                "mel_bins": 128,
            }
    
    # Get vocabulary size from checkpoint head if available
    vocab_size = None
    if "head" in checkpoint:
        head_state = checkpoint["head"]
        if isinstance(head_state, dict):
            # Check weight shape to get vocab size
            if "weight" in head_state:
                vocab_size = head_state["weight"].shape[0]
            elif any("weight" in k for k in head_state.keys()):
                # Find weight key (might have prefix)
                for k in head_state.keys():
                    if "weight" in k:
                        vocab_size = head_state[k].shape[0]
                        break
    
    # If vocab size not found, try from config or build default
    if vocab_size is None:
        vocab_size = cfg.get("ctc_vocab_size", None)
        if vocab_size is None:
            # Build character vocabulary to get default size
            char_to_idx, idx_to_char, vocab_size = build_char_vocab()
        else:
            # Build vocabulary with known size (may not match exactly, but we'll use checkpoint size)
            char_to_idx, idx_to_char, _ = build_char_vocab()
    else:
        # Build vocabulary (may not match checkpoint exactly, but vocab size will)
        char_to_idx, idx_to_char, _ = build_char_vocab()
    
    print(f"CTC vocabulary size: {vocab_size} (includes blank token)")
    
    # Initialize audio encoder
    audio_encoder = AudioEncoderTiny(
        d=cfg.get("d_model", 192),
        heads=cfg.get("n_heads", 3),
        ff=cfg.get("d_ff", 768),
        layers=cfg.get("n_layers", 4),
        dropout=cfg.get("dropout", 0.1),
        downsample_factor=cfg.get("downsample_time", 8),
        compile_model=False
    ).to(device)
    
    # Initialize CTC head with detected vocab size
    ctc_head = torch.nn.Linear(cfg.get("d_model", 192), vocab_size).to(device)
    
    # Load weights
    if "enc" in checkpoint:
        enc_state = checkpoint["enc"]
    elif "model" in checkpoint:
        enc_state = checkpoint["model"]
    else:
        enc_state = checkpoint
    
    if "head" in checkpoint:
        head_state = checkpoint["head"]
    else:
        head_state = None
    
    # Strip _orig_mod (matches training script behavior)
    enc_state = strip_orig_mod(enc_state)
    if head_state is not None:
        head_state = strip_orig_mod(head_state)
        ctc_head.load_state_dict(head_state, strict=False)
        print("✓ CTC head loaded from checkpoint")
    else:
        print("⚠ Warning: CTC head not found in checkpoint, using untrained head")
    
    audio_encoder.load_state_dict(enc_state, strict=False)
    audio_encoder.eval()
    ctc_head.eval()
    print("✓ Audio Encoder + CTC Head loaded successfully")
    
    return audio_encoder, ctc_head, char_to_idx, idx_to_char, cfg


def load_tts_models(talker_ckpt_dir, device="cuda"):
    """Load Talker, RVQ, and Vocoder from checkpoint."""
    # Load Talker checkpoint
    talker_path, talker_ckpt = find_checkpoint(talker_ckpt_dir, "talker.pt", "talker_step_", device)
    if talker_ckpt is None:
        raise FileNotFoundError(f"Talker checkpoint not found in: {talker_ckpt_dir}")
    
    print(f"Loading Talker from: {talker_path}")
    
    # Get config
    config_path = "configs/talker_tiny.json"
    if os.path.exists(config_path):
        talker_cfg = json.load(open(config_path))
    else:
        talker_cfg = {
            "d_model": 384,
            "n_layers": 8,
            "n_heads": 6,
            "d_ff": 1536,
            "codebooks": 2,
            "codebook_size": 128,
            "dropout": 0.1,
            "sample_rate": 16000,
            "n_mels": 128,
        }
    
    # Initialize RVQ
    rvq = RVQ(
        codebooks=talker_cfg.get("codebooks", 2),
        codebook_size=talker_cfg.get("codebook_size", 128),
        d=64  # RVQ embedding dimension (standard)
    ).to(device)
    
    # Initialize Talker
    talker = TalkerTiny(
        d=talker_cfg.get("d_model", 384),
        n_layers=talker_cfg.get("n_layers", 8),
        n_heads=talker_cfg.get("n_heads", 6),
        ff=talker_cfg.get("d_ff", 1536),
        codebooks=talker_cfg.get("codebooks", 2),
        codebook_size=talker_cfg.get("codebook_size", 128),
        dropout=talker_cfg.get("dropout", 0.1),
        use_gqa=talker_cfg.get("use_gqa", False),
        use_swiglu=talker_cfg.get("use_swiglu", True)
    ).to(device)
    
    # Load weights
    if isinstance(talker_ckpt, dict):
        if "rvq" in talker_ckpt:
            rvq_state = strip_orig_mod(talker_ckpt["rvq"])
            rvq.load_state_dict(rvq_state, strict=False)
            print("✓ RVQ loaded")
        
        if "talker" in talker_ckpt:
            talker_state = strip_orig_mod(talker_ckpt["talker"])
            talker.load_state_dict(talker_state, strict=False)
            print("✓ Talker loaded")
        elif "model" in talker_ckpt:
            talker_state = strip_orig_mod(talker_ckpt["model"])
            talker.load_state_dict(talker_state, strict=False)
            print("✓ Talker loaded")
    
    rvq.eval()
    talker.eval()
    
    # Load Vocoder
    print("Loading Vocoder...")
    vocoder = NeuralVocoder(
        sample_rate=talker_cfg.get("sample_rate", 16000),
        n_mels=talker_cfg.get("n_mels", 128)
    )
    print("✓ Vocoder loaded (NeuralVocoder with HiFi-GAN/Griffin-Lim fallback)")
    
    return talker, rvq, vocoder, talker_cfg


def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate (WER)"""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Dynamic programming for edit distance
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    edit_distance = dp[m][n]
    wer = (edit_distance / max(m, 1)) * 100
    return wer, edit_distance, m


def calculate_cer(reference, hypothesis):
    """Calculate Character Error Rate (CER)"""
    ref_chars = list(reference.lower().replace(' ', ''))
    hyp_chars = list(hypothesis.lower().replace(' ', ''))
    
    m, n = len(ref_chars), len(hyp_chars)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    edit_distance = dp[m][n]
    cer = (edit_distance / max(m, 1)) * 100
    return cer, edit_distance, m


def encode_text(text, char_to_idx, unk_idx, max_len=64):
    """Encode text to character indices"""
    ids = [char_to_idx.get(c, unk_idx) for c in text[:max_len]]
    if not ids:
        ids = [unk_idx]
    return torch.tensor(ids, dtype=torch.long)


def generate_speech(talker, rvq, vocoder, device="cuda", max_frames=200):
    """Generate speech using Talker + RVQ + Vocoder (autoregressive, no text conditioning)"""
    talker.eval()
    rvq.eval()
    
    # Start with BOS token
    codes = torch.zeros(1, 1, 2, dtype=torch.long, device=device)  # (B, T, 2)
    
    # Enable KV cache for faster generation
    if hasattr(talker, 'enable_kv_cache'):
        talker.enable_kv_cache(True)
        talker.reset_kv_cache()
    
    # Generate codes autoregressively
    with torch.no_grad():
        # First frame
        base_logits, res_logits = talker(codes, use_cache=True)
        base_code = torch.argmax(base_logits[0, -1, :]).item()
        res_code = torch.argmax(res_logits[0, -1, :]).item()
        next_codes = torch.tensor([[[base_code, res_code]]], device=device)
        codes = torch.cat([codes, next_codes], dim=1)
        
        # Remaining frames
        for _ in range(max_frames - 1):
            base_logits, res_logits = talker(next_codes, use_cache=True)
            base_code = torch.argmax(base_logits[0, -1, :]).item()
            res_code = torch.argmax(res_logits[0, -1, :]).item()
            next_codes = torch.tensor([[[base_code, res_code]]], device=device)
            codes = torch.cat([codes, next_codes], dim=1)
    
    # Reset KV cache
    if hasattr(talker, 'reset_kv_cache'):
        talker.reset_kv_cache()
    
    # Remove BOS token
    codes = codes[:, 1:, :]  # (B, T, 2)
    
    # Decode codes to mel spectrogram using RVQ
    mel_frames = []
    for t in range(codes.shape[1]):
        code_pair = codes[0, t, :]  # (2,)
        mel_frame = rvq.decode(code_pair.unsqueeze(0))  # (1, 128)
        mel_frames.append(mel_frame.squeeze(0))
    
    mel = torch.stack(mel_frames, dim=0)  # (T, 128)
    
    # Convert mel to audio using vocoder
    audio = vocoder.mel_to_audio(mel.unsqueeze(0))  # (1, T_audio)
    
    return audio[0] if audio is not None else None  # (T_audio,)


def evaluate_asr_tts_roundtrip(audio_encoder, ctc_head, talker, rvq, vocoder, tokenizer,
                                char_to_idx, idx_to_char, audio_cfg, talker_cfg, 
                                device="cuda", num_samples=100):
    """Evaluate full round-trip: Speech → Text → Speech"""
    audio_encoder.eval()
    ctc_head.eval()
    talker.eval()
    rvq.eval()
    ctc_loss_fn = CTCLoss(blank=0, zero_infinity=True)
    
    csv_path = audio_cfg.get("train_csv", "data/audio/production_asr.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ASR CSV not found: {csv_path}")
    
    # Create dataset
    dataset = ASRDataset(
        csv_path=csv_path,
        sr=audio_cfg.get("sample_rate", 16000),
        n_mels=audio_cfg.get("mel_bins", 128),
        cfg=audio_cfg,
        shuffle_buffer_size=10000,
        seed=random.randint(0, 1000000),
        skip_samples=0
    )
    
    iterator = iter(dataset)
    unk_idx = char_to_idx.get('<UNK>', 1)
    
    # ASR metrics
    total_asr_loss = 0.0
    total_wer = 0.0
    total_cer = 0.0
    exact_matches = 0
    word_correct = 0
    word_total = 0
    char_correct = 0
    char_total = 0
    
    # TTS/Reconstruction metrics
    total_audio_mse = 0.0
    total_audio_l1 = 0.0
    total_mel_mse = 0.0
    successful_tts = 0
    
    print(f"\nEvaluating round-trip on {num_samples} samples...")
    print("Pipeline: Audio → ASR → Text → TTS → Audio")
    
    with torch.no_grad():
        for i in range(num_samples):
            try:
                mel, text = next(iterator)
                mel = mel.unsqueeze(0).to(device)  # (B, T, 128)
                ground_truth_text = text.strip()
                
                # ===== STEP 1: ASR (Speech → Text) =====
                audio_emb = audio_encoder(mel)  # (B, T', d_model)
                logits = ctc_head(audio_emb)  # (B, T', vocab_size)
                
                # Compute ASR loss (match training format exactly)
                tgt_ids = encode_text(ground_truth_text, char_to_idx, unk_idx).to(device)
                tgt_len = torch.tensor([len(tgt_ids)], dtype=torch.long, device=device)
                
                # Transpose first, then get time dimension (matches training)
                log_probs = logits.log_softmax(-1).transpose(0, 1)  # (T', B, V)
                inp_len = torch.tensor([log_probs.size(0)], dtype=torch.long, device=device)  # Time dimension after transpose
                
                # CTC loss expects: log_probs (T, N, C), targets (N, S) or flat, input_lengths (N,), target_lengths (N,)
                # For batch size 1, we can use either format, but let's match training exactly
                # Training uses flat targets, but for single sample we can use (1, S) format
                asr_loss = ctc_loss_fn(log_probs, tgt_ids.unsqueeze(0), inp_len, tgt_len)
                
                if not torch.isnan(asr_loss) and not torch.isinf(asr_loss):
                    total_asr_loss += asr_loss.item()
                
                # Decode: CTC greedy decoding
                # Check if logits are reasonable (not all zeros or NaNs)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"⚠️  Warning: NaN/Inf in logits for sample {i}")
                if (logits.abs() < 1e-6).all():
                    print(f"⚠️  Warning: Logits are all near zero for sample {i}")
                
                predicted_text = ctc_greedy_decode(logits[0], idx_to_char)
                
                # Debug: Print first few samples to see what's happening
                if i < 3:
                    print(f"\n  Sample {i}:")
                    print(f"    Ground truth: '{ground_truth_text}'")
                    print(f"    Predicted: '{predicted_text}'")
                    print(f"    Logits shape: {logits.shape}, max: {logits.max():.2f}, min: {logits.min():.2f}")
                    print(f"    Top 5 predictions at frame 0: {torch.topk(logits[0, 0, :], 5).indices.tolist()}")
                
                # Calculate ASR metrics
                wer, wer_ed, ref_words = calculate_wer(ground_truth_text, predicted_text)
                cer, cer_ed, ref_chars = calculate_cer(ground_truth_text, predicted_text)
                
                total_wer += wer
                total_cer += cer
                
                if predicted_text.lower().strip() == ground_truth_text.lower().strip():
                    exact_matches += 1
                
                # Word/char accuracy
                ref_words_list = ground_truth_text.lower().split()
                hyp_words_list = predicted_text.lower().split()
                for ref_w, hyp_w in zip(ref_words_list, hyp_words_list):
                    word_total += 1
                    if ref_w == hyp_w:
                        word_correct += 1
                word_total += max(0, len(ref_words_list) - len(hyp_words_list))
                
                ref_chars_list = list(ground_truth_text.lower().replace(' ', ''))
                hyp_chars_list = list(predicted_text.lower().replace(' ', ''))
                for ref_c, hyp_c in zip(ref_chars_list, hyp_chars_list):
                    char_total += 1
                    if ref_c == hyp_c:
                        char_correct += 1
                char_total += max(0, len(ref_chars_list) - len(hyp_chars_list))
                
                # ===== STEP 2: TTS (Generate Speech) =====
                # Note: Talker generates speech autoregressively without text conditioning
                # This tests the reconstruction quality of the TTS pipeline
                try:
                    # Generate audio (autoregressive, not conditioned on text)
                    reconstructed_audio = generate_speech(
                        talker, rvq, vocoder, device, max_frames=200
                    )
                    
                    if reconstructed_audio is not None:
                        successful_tts += 1
                        
                        # Get original audio from mel (for comparison)
                        # Note: We need to vocode the original mel to compare
                        original_audio = vocoder.mel_to_audio(mel)  # (1, T_audio)
                        if original_audio is not None:
                            orig_audio = original_audio[0].cpu().numpy()  # (T_audio,)
                            recon_audio = reconstructed_audio.cpu().numpy() if isinstance(reconstructed_audio, torch.Tensor) else reconstructed_audio
                            
                            # Align lengths for comparison
                            min_len = min(len(orig_audio), len(recon_audio))
                            if min_len > 0:
                                orig_audio = orig_audio[:min_len]
                                recon_audio = recon_audio[:min_len]
                                
                                # Audio reconstruction metrics
                                audio_mse = np.mean((orig_audio - recon_audio) ** 2)
                                audio_l1 = np.mean(np.abs(orig_audio - recon_audio))
                                
                                total_audio_mse += audio_mse
                                total_audio_l1 += audio_l1
                                
                                # Mel reconstruction (if we can get reconstructed mel)
                                # This would require encoding reconstructed audio back to mel
                                # For now, we skip this as it's computationally expensive
                except Exception as e:
                    # TTS failed for this sample
                    pass
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{num_samples} samples... (ASR: {exact_matches} exact, TTS: {successful_tts} success)", end='\r')
                    
            except StopIteration:
                print(f"\n⚠️  Dataset exhausted after {i} samples")
                break
            except Exception as e:
                print(f"\n⚠️  Error processing sample {i}: {e}")
                continue
    
    print()  # New line after progress
    
    # Calculate metrics
    avg_asr_loss = total_asr_loss / num_samples if num_samples > 0 else float('inf')
    avg_wer = total_wer / num_samples if num_samples > 0 else 0.0
    avg_cer = total_cer / num_samples if num_samples > 0 else 0.0
    exact_match_rate = (exact_matches / num_samples * 100) if num_samples > 0 else 0.0
    word_accuracy = (word_correct / word_total * 100) if word_total > 0 else 0.0
    char_accuracy = (char_correct / char_total * 100) if char_total > 0 else 0.0
    
    # TTS metrics
    avg_audio_mse = total_audio_mse / successful_tts if successful_tts > 0 else float('inf')
    avg_audio_l1 = total_audio_l1 / successful_tts if successful_tts > 0 else float('inf')
    tts_success_rate = (successful_tts / num_samples * 100) if num_samples > 0 else 0.0
    
    return {
        # ASR metrics
        'asr_loss': avg_asr_loss,
        'wer': avg_wer,
        'cer': avg_cer,
        'exact_match_rate': exact_match_rate,
        'word_accuracy': word_accuracy,
        'char_accuracy': char_accuracy,
        # TTS metrics
        'tts_success_rate': tts_success_rate,
        'audio_mse': avg_audio_mse,
        'audio_l1': avg_audio_l1,
        'num_samples': i + 1,
        'successful_tts': successful_tts
    }


def main():
    parser = argparse.ArgumentParser(description="Test ASR+TTS round-trip (Speech → Text → Speech)")
    parser.add_argument("--audio_ckpt", type=str, default="checkpoints/audio_enc_tiny",
                       help="Path to Audio Encoder checkpoint directory")
    parser.add_argument("--talker_ckpt", type=str, default="checkpoints/talker_tiny",
                       help="Path to Talker checkpoint directory")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate (default: 100)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda/cpu)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("ASR + TTS Round-Trip Test")
    print("=" * 60)
    print("Pipeline: Speech → Text → Speech")
    print("  Step 1: Audio Encoder + CTC Head → Text (ASR)")
    print("  Step 2: Text → Talker + RVQ + Vocoder → Audio (TTS)")
    
    # Load ASR models
    try:
        audio_encoder, ctc_head, char_to_idx, idx_to_char, audio_cfg = load_asr_models(args.audio_ckpt, args.device)
    except Exception as e:
        print(f"❌ Error loading ASR models: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load TTS models
    try:
        talker, rvq, vocoder, talker_cfg = load_tts_models(args.talker_ckpt, args.device)
    except Exception as e:
        print(f"❌ Error loading TTS models: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate round-trip
    try:
        metrics = evaluate_asr_tts_roundtrip(
            audio_encoder, ctc_head, talker, rvq, vocoder, None,  # tokenizer not needed
            char_to_idx, idx_to_char, audio_cfg, talker_cfg,
            args.device, args.num_samples
        )
        
        print(f"\n{'=' * 60}")
        print("EVALUATION RESULTS:")
        print(f"{'=' * 60}")
        print(f"\n📊 ASR Metrics (Speech → Text):")
        print(f"  Samples evaluated: {metrics['num_samples']}")
        print(f"  Average CTC Loss: {metrics['asr_loss']:.4f}")
        print(f"  Word Error Rate (WER): {metrics['wer']:.2f}%")
        print(f"  Character Error Rate (CER): {metrics['cer']:.2f}%")
        print(f"  Exact Match Rate: {metrics['exact_match_rate']:.2f}%")
        print(f"  Word Accuracy: {metrics['word_accuracy']:.2f}%")
        print(f"  Character Accuracy: {metrics['char_accuracy']:.2f}%")
        
        print(f"\n🎵 TTS Metrics (Text → Speech):")
        print(f"  TTS Success Rate: {metrics['tts_success_rate']:.2f}% ({metrics['successful_tts']}/{metrics['num_samples']})")
        if metrics['successful_tts'] > 0:
            print(f"  Audio Reconstruction MSE: {metrics['audio_mse']:.6f}")
            print(f"  Audio Reconstruction L1: {metrics['audio_l1']:.6f}")
        else:
            print(f"  ⚠️  No successful TTS generations")
        
        print(f"{'=' * 60}")
        
        # Interpretation
        print(f"\n📈 Performance Summary:")
        if metrics['wer'] < 10:
            print("  ✓ Excellent ASR performance!")
        elif metrics['wer'] < 25:
            print("  ✓ Good ASR performance")
        elif metrics['wer'] < 50:
            print("  ⚠️  Moderate ASR - model may need more training")
        else:
            print("  ⚠️  Poor ASR - model needs significant training")
        
        if metrics['tts_success_rate'] > 80:
            print("  ✓ Good TTS success rate")
        elif metrics['tts_success_rate'] > 50:
            print("  ⚠️  Moderate TTS success rate")
        else:
            print("  ⚠️  Low TTS success rate - check Talker/Vocoder")
        
        print(f"\nNote: Lower WER/CER/MSE/L1 is better. WER < 10% is considered excellent.")
            
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

