"""
Complete test script to validate Audio Encoder (ASR) accuracy.
Measures CER, WER, and provides detailed transcription examples.
"""

import torch
import torch.nn as nn
import json
import os
import argparse
import random
import torchaudio  # Only used for transforms, not for loading audio
from omni.audio_encoder import AudioEncoderTiny
from omni.utils import ASRDataset, load_audio, find_checkpoint, strip_orig_mod
from tqdm import tqdm

try:
    import Levenshtein
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False
    print("⚠️  Warning: python-Levenshtein not installed. Install with: pip install python-Levenshtein")
    print("   Falling back to basic error rate calculation.")


def compute_cer_basic(pred, target):
    """Basic Character Error Rate without Levenshtein."""
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    
    # Simple character-level comparison
    matches = sum(1 for p, t in zip(pred, target) if p == t)
    max_len = max(len(pred), len(target))
    return 1.0 - (matches / max_len)


def compute_wer_basic(pred, target):
    """Basic Word Error Rate without Levenshtein."""
    pred_words = pred.split()
    target_words = target.split()
    
    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    
    # Simple word-level comparison
    matches = sum(1 for p, t in zip(pred_words, target_words) if p == t)
    max_len = max(len(pred_words), len(target_words))
    return 1.0 - (matches / max_len)


def compute_cer(pred, target):
    """Character Error Rate using Levenshtein distance."""
    if HAS_LEVENSHTEIN:
        return Levenshtein.distance(pred, target) / max(len(target), 1)
    else:
        return compute_cer_basic(pred, target)


def compute_wer(pred, target):
    """Word Error Rate using Levenshtein distance."""
    if HAS_LEVENSHTEIN:
        pred_words = pred.split()
        target_words = target.split()
        return Levenshtein.distance(pred_words, target_words) / max(len(target_words), 1)
    else:
        return compute_wer_basic(pred, target)


def decode_ctc_greedy(logits, idx_to_char, blank_idx=0):
    """
    Greedy CTC decoder (argmax at each timestep).
    
    Args:
        logits: (B, T, vocab_size) - model outputs
        idx_to_char: dict mapping indices to characters
        blank_idx: index of blank token (default: 0)
    
    Returns:
        List of decoded strings
    """
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)  # (B, T)
    
    decoded = []
    for pred_seq in preds:
        chars = []
        prev = None
        for idx in pred_seq:
            idx = idx.item()
            # Skip blanks and repeated characters (CTC collapsing)
            if idx != blank_idx and idx != prev:
                if idx in idx_to_char:
                    char = idx_to_char[idx]
                    # Skip special tokens
                    if char not in ['<BLANK>', '<UNK>']:
                        chars.append(char)
            prev = idx
        decoded.append(''.join(chars))
    
    return decoded


def decode_ctc_beam_search(logits, idx_to_char, blank_idx=0, beam_width=10, alpha=0.3, beta=1.2, lm_scorer=None):
    """
    Beam search CTC decoder with optional language model integration.
    
    This implementation uses a prefix beam search algorithm that maintains
    multiple hypotheses (beams) at each timestep and selects the most probable
    sequences based on CTC probabilities and optional language model scores.
    
    Args:
        logits: (B, T, vocab_size) - model outputs (log probabilities)
        idx_to_char: dict mapping indices to characters
        blank_idx: index of blank token (default: 0)
        beam_width: number of beams to maintain (default: 10)
        alpha: language model weight (0.0 = no LM, higher = more LM influence)
        beta: length normalization factor (encourages longer sequences)
        lm_scorer: optional language model scorer function (text -> score)
    
    Returns:
        List of decoded strings (one per batch item)
    
    Algorithm:
        For each timestep:
        1. For each beam, extend with all possible characters
        2. Track two probabilities per prefix:
           - p_blank: probability ending in blank
           - p_non_blank: probability ending in non-blank
        3. Combine paths that produce same prefix
        4. Apply optional language model scoring
        5. Keep top beam_width candidates
    """
    probs = torch.softmax(logits, dim=-1)  # (B, T, vocab_size)
    batch_size, T, vocab_size = probs.shape
    
    decoded = []
    
    for b in range(batch_size):
        batch_probs = probs[b]  # (T, vocab_size)
        
        # Initialize beams
        # Each beam is: (prefix_text, p_blank, p_non_blank, score)
        beams = [("", 1.0, 0.0, 0.0)]  # (text, p_b, p_nb, lm_score)
        
        for t in range(T):
            new_beams = {}  # Dict[prefix] -> (p_blank, p_non_blank, lm_score)
            
            for prefix, p_b, p_nb, lm_score in beams:
                # Get probability at this timestep
                prob_t = batch_probs[t]  # (vocab_size,)
                
                # Option 1: Extend with blank
                p_blank_next = prob_t[blank_idx].item()
                total_p = (p_b + p_nb) * p_blank_next
                
                if prefix not in new_beams:
                    new_beams[prefix] = (total_p, 0.0, lm_score)
                else:
                    pb_old, pnb_old, lm_old = new_beams[prefix]
                    new_beams[prefix] = (pb_old + total_p, pnb_old, lm_old)
                
                # Option 2: Extend with character
                for idx in range(vocab_size):
                    if idx == blank_idx or idx not in idx_to_char:
                        continue
                    
                    char = idx_to_char[idx]
                    if char in ['<BLANK>', '<UNK>']:
                        continue
                    
                    new_prefix = prefix + char
                    p_char = prob_t[idx].item()
                    
                    # If extending with same character as last
                    if len(prefix) > 0 and prefix[-1] == char:
                        # Can only extend from blank state (CTC rule)
                        total_p = p_b * p_char
                    else:
                        # Can extend from either state
                        total_p = (p_b + p_nb) * p_char
                    
                    # Calculate LM score for new prefix if LM available
                    new_lm_score = lm_score
                    if lm_scorer is not None:
                        new_lm_score = lm_scorer(new_prefix)
                    
                    if new_prefix not in new_beams:
                        new_beams[new_prefix] = (0.0, total_p, new_lm_score)
                    else:
                        pb_old, pnb_old, lm_old = new_beams[new_prefix]
                        new_beams[new_prefix] = (pb_old, pnb_old + total_p, new_lm_score)
            
            # Prune beams - keep top beam_width
            # Combined score: log(CTC_prob) + alpha * LM_score + beta * len(prefix)
            scored_beams = []
            for prefix, (pb, pnb, lm_score) in new_beams.items():
                total_p = pb + pnb
                if total_p > 0:
                    # Log probability + LM score + length bonus
                    score = (
                        torch.log(torch.tensor(total_p)).item() + 
                        alpha * lm_score + 
                        beta * len(prefix)
                    )
                    scored_beams.append((prefix, pb, pnb, lm_score, score))
            
            # Sort by score and keep top beams
            scored_beams.sort(key=lambda x: x[4], reverse=True)
            beams = [(prefix, pb, pnb, lm_score) for prefix, pb, pnb, lm_score, _ in scored_beams[:beam_width]]
            
            # If no beams left, restart with empty
            if not beams:
                beams = [("", 1.0, 0.0, 0.0)]
        
        # Select best beam
        best_beam = max(beams, key=lambda x: x[1] + x[2])
        decoded.append(best_beam[0])
    
    return decoded


def load_model_and_head(checkpoint_dir, device="cuda"):
    """Load Audio Encoder model and CTC head from checkpoint."""
    checkpoint_path, checkpoint = find_checkpoint(checkpoint_dir, "audio_enc.pt", "audio_enc_step_", device)
    if checkpoint is None:
        raise FileNotFoundError(f"Checkpoint not found in: {checkpoint_dir}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Get config from checkpoint or load from config file
    if "config" in checkpoint:
        cfg = checkpoint["config"]
    else:
        # Load config from JSON file based on checkpoint directory
        checkpoint_name = os.path.basename(checkpoint_dir)
        config_path = f"configs/{checkpoint_name}.json"
        
        if os.path.exists(config_path):
            print(f"Loading config from: {config_path}")
            with open(config_path, 'r') as f:
                cfg = json.load(f)
        else:
            raise FileNotFoundError(
                f"Config not found in checkpoint and config file not found: {config_path}"
            )
    
    # Initialize model
    model = AudioEncoderTiny(
        d=cfg.get("d_model", 192),
        heads=cfg.get("n_heads", 3),
        ff=cfg.get("d_ff", 768),
        layers=cfg.get("n_layers", 4),
        dropout=cfg.get("dropout", 0.1),
        downsample_factor=cfg.get("downsample_time", 8),
        compile_model=False,
        use_spiking=cfg.get("use_spiking", False),
        use_ltc=cfg.get("use_ltc", False)
    ).to(device)
    
    # Load model weights
    if "enc" in checkpoint:
        state_dict = checkpoint["enc"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    state_dict = strip_orig_mod(state_dict)
    model.load_state_dict(state_dict, strict=False)
    
    # Load metadata for vocabulary
    metadata_path = os.path.join(checkpoint_dir, "audio_enc_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Get vocabulary mappings
        char_to_idx = metadata.get("char_to_idx", {})
        idx_to_char = metadata.get("idx_to_char", {})
        vocab_size = metadata.get("vocab_size", len(char_to_idx))
        
        # Convert string keys back to integers for idx_to_char (JSON saves all keys as strings)
        if idx_to_char and isinstance(list(idx_to_char.keys())[0], str):
            idx_to_char = {int(k): v for k, v in idx_to_char.items()}
        
        print(f"  Loaded vocabulary: {vocab_size} characters")
        print(f"  Sample mappings: {list(idx_to_char.items())[:5]}")
    else:
        print("⚠️  Warning: Metadata file not found. Using defaults.")
        vocab_size = cfg.get("ctc_vocab_size", 256)
        char_to_idx = {}
        idx_to_char = {}
    
    # Initialize CTC head
    d_model = cfg.get("d_model", 192)
    head = nn.Linear(d_model, vocab_size).to(device)
    
    # Load head weights
    if "head" in checkpoint:
        head_state_dict = checkpoint["head"]
        head_state_dict = strip_orig_mod(head_state_dict)
        head.load_state_dict(head_state_dict, strict=False)
    
    model.eval()
    head.eval()
    
    print("✓ Model and head loaded successfully")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Characters in vocab: {len(char_to_idx)}")
    
    return model, head, idx_to_char, cfg


def evaluate_accuracy(model, head, idx_to_char, cfg, device="cuda", num_samples=100, verbose=True, 
                     use_beam_search=True, beam_width=10):
    """
    Evaluate ASR accuracy with CER and WER metrics.
    
    Args:
        model: AudioEncoderTiny model
        head: CTC head (Linear layer)
        idx_to_char: dictionary mapping indices to characters
        cfg: configuration dictionary
        device: device to run on
        num_samples: number of samples to evaluate
        verbose: whether to print progress
        use_beam_search: whether to use beam search (default: True) vs greedy
        beam_width: beam width for beam search (default: 10)
    
    Returns:
        Dictionary with metrics and examples
    """
    model.eval()
    head.eval()
    
    csv_path = cfg.get("train_csv", "data/audio/production_asr.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ASR CSV not found: {csv_path}")
    
    # Create dataset
    dataset = ASRDataset(
        csv_path=csv_path,
        sr=cfg.get("sample_rate", 16000),
        n_mels=cfg.get("mel_bins", 128),
        cfg=cfg,
        shuffle_buffer_size=10000,
        seed=42,  # Fixed seed for reproducibility
        skip_samples=0
    )
    
    total_cer = 0.0
    total_wer = 0.0
    total_cer_greedy = 0.0
    total_wer_greedy = 0.0
    num_valid = 0
    examples = []
    
    decoder_name = f"Beam Search (width={beam_width})" if use_beam_search else "Greedy"
    
    if verbose:
        print(f"\nEvaluating on {num_samples} samples using {decoder_name} decoder...")
        iterator = tqdm(dataset, total=num_samples, desc="Evaluating")
    else:
        iterator = iter(dataset)
    
    with torch.no_grad():
        for i, (mel, text) in enumerate(iterator):
            if i >= num_samples:
                break
            
            try:
                mel = mel.unsqueeze(0).to(device)  # Add batch dimension
                
                # Forward pass
                x = model(mel)  # (1, T', d_model)
                logits = head(x)  # (1, T', vocab_size)
                
                # Decode with primary decoder
                if use_beam_search:
                    pred_texts = decode_ctc_beam_search(logits, idx_to_char, beam_width=beam_width)
                else:
                    pred_texts = decode_ctc_greedy(logits, idx_to_char)
                pred_text = pred_texts[0]
                
                # Also decode with greedy for comparison
                greedy_texts = decode_ctc_greedy(logits, idx_to_char)
                greedy_text = greedy_texts[0]
                
                # Normalize for comparison (lowercase, strip whitespace)
                pred_normalized = pred_text.lower().strip()
                greedy_normalized = greedy_text.lower().strip()
                target_normalized = text.lower().strip()
                
                # Compute metrics for primary decoder
                cer = compute_cer(pred_normalized, target_normalized)
                wer = compute_wer(pred_normalized, target_normalized)
                
                # Compute metrics for greedy decoder
                cer_greedy = compute_cer(greedy_normalized, target_normalized)
                wer_greedy = compute_wer(greedy_normalized, target_normalized)
                
                total_cer += cer
                total_wer += wer
                total_cer_greedy += cer_greedy
                total_wer_greedy += wer_greedy
                num_valid += 1
                
                # Save first 10 examples
                if len(examples) < 10:
                    examples.append({
                        'target': text,
                        'predicted': pred_text,
                        'greedy': greedy_text,
                        'cer': cer,
                        'wer': wer,
                        'cer_greedy': cer_greedy,
                        'wer_greedy': wer_greedy
                    })
                
            except Exception as e:
                if verbose:
                    print(f"\n⚠️  Error processing sample {i}: {e}")
                continue
    
    if num_valid == 0:
        raise ValueError("No valid samples processed!")
    
    # Calculate average metrics
    avg_cer = total_cer / num_valid
    avg_wer = total_wer / num_valid
    avg_cer_greedy = total_cer_greedy / num_valid
    avg_wer_greedy = total_wer_greedy / num_valid
    
    return {
        'cer': avg_cer,
        'wer': avg_wer,
        'cer_greedy': avg_cer_greedy,
        'wer_greedy': avg_wer_greedy,
        'num_samples': num_valid,
        'examples': examples,
        'use_beam_search': use_beam_search,
        'beam_width': beam_width if use_beam_search else None
    }


def print_results(metrics):
    """Pretty print evaluation results."""
    print(f"\n{'='*70}")
    print("ACCURACY EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Samples Evaluated: {metrics['num_samples']}")
    
    # Display decoder info
    if metrics.get('use_beam_search'):
        print(f"Primary Decoder: Beam Search (width={metrics.get('beam_width', 10)})")
    else:
        print(f"Primary Decoder: Greedy")
    
    print(f"\n{'-'*70}")
    print("PRIMARY DECODER PERFORMANCE:")
    print(f"{'-'*70}")
    print(f"Character Error Rate (CER): {metrics['cer']*100:.2f}%")
    print(f"Word Error Rate (WER): {metrics['wer']*100:.2f}%")
    
    # Show greedy comparison if using beam search
    if metrics.get('use_beam_search'):
        print(f"\n{'-'*70}")
        print("GREEDY DECODER COMPARISON:")
        print(f"{'-'*70}")
        print(f"Character Error Rate (CER): {metrics['cer_greedy']*100:.2f}%")
        print(f"Word Error Rate (WER): {metrics['wer_greedy']*100:.2f}%")
        
        # Calculate improvement
        cer_improvement = (metrics['cer_greedy'] - metrics['cer']) * 100
        wer_improvement = (metrics['wer_greedy'] - metrics['wer']) * 100
        
        print(f"\n{'-'*70}")
        print("BEAM SEARCH IMPROVEMENT:")
        print(f"{'-'*70}")
        print(f"CER Improvement: {cer_improvement:+.2f}% {'✓' if cer_improvement > 0 else '✗'}")
        print(f"WER Improvement: {wer_improvement:+.2f}% {'✓' if wer_improvement > 0 else '✗'}")
    
    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print(f"{'='*70}")
    
    if metrics['cer'] < 0.05:
        cer_status = "✓ EXCELLENT - Production ready!"
    elif metrics['cer'] < 0.10:
        cer_status = "✓ GOOD - Working well"
    elif metrics['cer'] < 0.20:
        cer_status = "⚠ ACCEPTABLE - May need improvement"
    else:
        cer_status = "✗ POOR - Needs more training"
    
    if metrics['wer'] < 0.10:
        wer_status = "✓ EXCELLENT - Production ready!"
    elif metrics['wer'] < 0.20:
        wer_status = "✓ GOOD - Working well"
    elif metrics['wer'] < 0.30:
        wer_status = "⚠ ACCEPTABLE - May need improvement"
    else:
        wer_status = "✗ POOR - Needs more training"
    
    print(f"CER Assessment: {cer_status}")
    print(f"WER Assessment: {wer_status}")
    
    # Print examples
    if metrics['examples']:
        print(f"\n{'='*70}")
        print("SAMPLE TRANSCRIPTIONS:")
        print(f"{'='*70}")
        
        for idx, ex in enumerate(metrics['examples']):
            print(f"\nExample {idx+1}:")
            print(f"  Target:    '{ex['target']}'")
            print(f"  Predicted: '{ex['predicted']}'")
            if 'greedy' in ex and ex['greedy'] != ex['predicted']:
                print(f"  Greedy:    '{ex['greedy']}'")
            print(f"  CER: {ex['cer']*100:.1f}% | WER: {ex['wer']*100:.1f}%")
            if 'cer_greedy' in ex:
                print(f"  (Greedy CER: {ex['cer_greedy']*100:.1f}% | Greedy WER: {ex['wer_greedy']*100:.1f}%)")
    
    print(f"\n{'='*70}")


def test_single_audio(model, head, idx_to_char, audio_path, cfg, device="cuda", use_beam_search=True, beam_width=10):
    """Test on a single audio file."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    print(f"\nTesting single audio file: {audio_path}")
    
    # Load audio
    wav, orig_sr = load_audio(audio_path)
    sr = cfg.get("sample_rate", 16000)
    
    # Resample if needed
    if orig_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_sr, sr)
        wav = resampler(wav)
    
    # Convert to mono if stereo
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    
    # Create mel spectrogram
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        hop_length=160,
        win_length=400,
        n_mels=cfg.get("mel_bins", 128)
    ).to(device)
    
    wav = wav.to(device)
    mel = mel_spec(wav)[0].T.unsqueeze(0)  # (1, T, 128)
    
    # Forward pass
    model.eval()
    head.eval()
    
    with torch.no_grad():
        x = model(mel)
        logits = head(x)
        
        # Decode with both methods
        if use_beam_search:
            pred_texts = decode_ctc_beam_search(logits, idx_to_char, beam_width=beam_width)
            print(f"\nBeam Search Transcription (width={beam_width}): '{pred_texts[0]}'")
        
        greedy_texts = decode_ctc_greedy(logits, idx_to_char)
        print(f"Greedy Transcription: '{greedy_texts[0]}'")
    
    return pred_texts[0] if use_beam_search else greedy_texts[0]


def main():
    parser = argparse.ArgumentParser(description="Test Audio Encoder (ASR) with accuracy metrics")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/audio_enc_tiny",
                       help="Path to Audio Encoder checkpoint directory")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate (default: 100)")
    parser.add_argument("--audio", type=str, default=None,
                       help="Path to single audio file to test (optional)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with 10 samples")
    parser.add_argument("--greedy", action="store_true",
                       help="Use greedy decoding instead of beam search")
    parser.add_argument("--beam_width", type=int, default=10,
                       help="Beam width for beam search (default: 10)")
    args = parser.parse_args()
    
    if args.quick:
        args.num_samples = 10
    
    print("=" * 70)
    print("AUDIO ENCODER (ASR) ACCURACY TEST")
    print("=" * 70)
    
    # Load model
    try:
        model, head, idx_to_char, cfg = load_model_and_head(args.checkpoint, args.device)
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test single audio if provided
    if args.audio:
        try:
            test_single_audio(
                model, head, idx_to_char, args.audio, cfg, args.device,
                use_beam_search=not args.greedy,
                beam_width=args.beam_width
            )
        except Exception as e:
            print(f"✗ Error testing audio file: {e}")
            import traceback
            traceback.print_exc()
        return
    
    # Evaluate on dataset
    try:
        metrics = evaluate_accuracy(
            model, head, idx_to_char, cfg, 
            device=args.device, 
            num_samples=args.num_samples,
            verbose=True,
            use_beam_search=not args.greedy,
            beam_width=args.beam_width
        )
        
        print_results(metrics)
        
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()