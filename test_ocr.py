"""
Complete test script to validate OCR model accuracy.
Measures CER, WER, exact match rate, and provides detailed analysis.
"""

import torch
from torch.nn import CrossEntropyLoss
import json
import os
import argparse
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from omni.ocr_model import OCRModel
from omni.utils import OCRDataset, find_checkpoint, strip_orig_mod
from tqdm import tqdm

# Try to import Levenshtein for better edit distance
try:
    import Levenshtein
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False
    print("⚠️  python-Levenshtein not installed. Using fallback edit distance.")


def levenshtein_distance(s1, s2):
    """Calculate Levenshtein edit distance between two strings."""
    if HAS_LEVENSHTEIN:
        return Levenshtein.distance(s1, s2)
    
    # Fallback implementation
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]


def compute_cer(pred, target):
    """
    Compute Character Error Rate (CER).
    CER = edit_distance / len(target)
    """
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    
    edit_dist = levenshtein_distance(pred, target)
    return edit_dist / len(target)


def compute_wer(pred, target):
    """
    Compute Word Error Rate (WER).
    WER = edit_distance(words) / num_target_words
    """
    pred_words = pred.split()
    target_words = target.split()
    
    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    
    if HAS_LEVENSHTEIN:
        edit_dist = Levenshtein.distance(pred_words, target_words)
    else:
        # Fallback: word-level edit distance
        edit_dist = levenshtein_distance(pred_words, target_words)
    
    return edit_dist / len(target_words)


def load_model_and_vocab(checkpoint_dir, device="cuda"):
    """Load OCR model and vocabulary from checkpoint."""
    checkpoint_path, checkpoint = find_checkpoint(checkpoint_dir, "ocr.pt", "ocr_step_", device)
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
    
    # Get vocabulary - try checkpoint first, then metadata file
    if "char_to_idx" in checkpoint and "idx_to_char" in checkpoint:
        char_to_idx = checkpoint["char_to_idx"]
        idx_to_char = checkpoint["idx_to_char"]
        # Ensure idx_to_char has integer keys (in case it was loaded from JSON)
        if idx_to_char and isinstance(next(iter(idx_to_char.keys())), str):
            idx_to_char = {int(k): v for k, v in idx_to_char.items()}
        vocab_size = len(char_to_idx)
    else:
        # Try loading from metadata file
        metadata_path = os.path.join(checkpoint_dir, "ocr_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                char_to_idx = metadata.get("char_to_idx", {})
                idx_to_char = metadata.get("idx_to_char", {})
                # Convert string keys back to integers for idx_to_char (JSON keys are always strings)
                idx_to_char = {int(k): v for k, v in idx_to_char.items()}
                vocab_size = len(char_to_idx)
        else:
            raise ValueError("Checkpoint missing vocabulary. Cannot decode text.")
    
    print(f"✓ Loaded vocabulary (size: {vocab_size})")
    
    # Initialize model
    model = OCRModel(
        img_size=cfg.get("img_size", 224),
        patch=cfg.get("patch", 16),
        vision_d_model=cfg.get("vision_d_model", 128),
        vision_layers=cfg.get("vision_layers", 4),
        vision_heads=cfg.get("vision_heads", 2),
        vision_d_ff=cfg.get("vision_d_ff", 512),
        decoder_d_model=cfg.get("decoder_d_model", 256),
        decoder_layers=cfg.get("decoder_layers", 4),
        decoder_heads=cfg.get("decoder_heads", 4),
        decoder_d_ff=cfg.get("decoder_d_ff", 1024),
        vocab_size=vocab_size,
        dropout=cfg.get("dropout", 0.1),
        use_gqa=cfg.get("use_gqa", False),
        use_swiglu=cfg.get("use_swiglu", True),
        use_flash=cfg.get("use_flash", True),
        compile_model=False
    ).to(device)
    
    # Load weights
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    state_dict = strip_orig_mod(state_dict)
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    
    print("✓ Model loaded successfully")
    print(f"  Image size: {cfg.get('img_size', 224)}")
    print(f"  Decoder layers: {cfg.get('decoder_layers', 4)}")
    
    return model, char_to_idx, idx_to_char, cfg


def decode_greedy(model, image_tensor, char_to_idx, idx_to_char, device="cuda", max_length=256):
    """
    Greedy decoding: generate text from image autoregressively.
    
    Returns:
        Decoded text string
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # Start with BOS token
        bos_id = char_to_idx.get('<BOS>', 1)
        current_ids = torch.tensor([[bos_id]], device=device)
        
        generated_text = []
        
        for _ in range(max_length):
            # Forward pass
            logits = model(image_tensor, current_ids)  # (1, T, vocab_size)
            
            # Get next token (greedy)
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            
            # Check for EOS or special tokens
            if next_token_id in idx_to_char:
                char = idx_to_char[next_token_id]
                if char == '<EOS>':
                    break
                if char not in ['<PAD>', '<BOS>', '<UNK>']:
                    generated_text.append(char)
            
            # Append to sequence
            current_ids = torch.cat([current_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
    
    return ''.join(generated_text)


def decode_beam_search(model, image_tensor, char_to_idx, idx_to_char, device="cuda", max_length=256, beam_width=5):
    """
    Beam search decoding for better quality (optional, slower).
    
    Returns:
        Best decoded text string
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    bos_id = char_to_idx.get('<BOS>', 1)
    eos_id = char_to_idx.get('<EOS>', 2)
    
    # Initialize beam: list of (sequence, score)
    beams = [([bos_id], 0.0)]
    
    with torch.no_grad():
        for _ in range(max_length):
            new_beams = []
            
            for seq, score in beams:
                # Check if sequence ended
                if seq[-1] == eos_id:
                    new_beams.append((seq, score))
                    continue
                
                # Forward pass
                input_ids = torch.tensor([seq], device=device)
                logits = model(image_tensor, input_ids)  # (1, T, vocab_size)
                
                # Get log probabilities for next token
                next_token_logprobs = torch.log_softmax(logits[0, -1, :], dim=-1)
                
                # Get top-k candidates
                topk_logprobs, topk_ids = torch.topk(next_token_logprobs, beam_width)
                
                for logprob, token_id in zip(topk_logprobs, topk_ids):
                    new_seq = seq + [token_id.item()]
                    new_score = score + logprob.item()
                    new_beams.append((new_seq, new_score))
            
            # Keep top beam_width beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # Stop if all beams ended
            if all(seq[-1] == eos_id for seq, _ in beams):
                break
    
    # Get best beam
    best_seq, _ = beams[0]
    
    # Decode to text
    generated_text = []
    for token_id in best_seq:
        if token_id in idx_to_char:
            char = idx_to_char[token_id]
            if char == '<EOS>':
                break
            if char not in ['<PAD>', '<BOS>', '<UNK>']:
                generated_text.append(char)
    
    return ''.join(generated_text)


def evaluate_ocr_accuracy(model, char_to_idx, idx_to_char, cfg, device="cuda", num_samples=100, use_beam_search=False, verbose=True):
    """
    Comprehensive OCR evaluation.
    
    Metrics:
    - Character Error Rate (CER)
    - Word Error Rate (WER)
    - Exact Match Rate
    - Average edit distance
    - Per-character accuracy
    """
    model.eval()
    
    csv_path = cfg.get("train_csv", "data/ocr/ocr_train.csv")
    image_root = cfg.get("image_root", "data/ocr")
    img_size = cfg.get("img_size", 224)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"OCR CSV not found: {csv_path}")
    
    # Create dataset
    dataset = OCRDataset(
        csv_path=csv_path,
        image_root=image_root,
        img_size=img_size,
        cfg=cfg,
        shuffle_buffer_size=10000,
        seed=42,  # Fixed seed for reproducibility
        skip_samples=0,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char
    )
    
    # Accumulators
    total_cer = 0.0
    total_wer = 0.0
    exact_matches = 0
    total_edit_distance = 0
    char_correct = 0
    char_total = 0
    num_valid = 0
    examples = []
    
    if verbose:
        print(f"\nEvaluating OCR on {num_samples} samples...")
        if use_beam_search:
            print("  Using beam search decoding (slower, better quality)")
        else:
            print("  Using greedy decoding (faster)")
        iterator = tqdm(dataset, total=num_samples, desc="Processing")
    else:
        iterator = iter(dataset)
    
    with torch.no_grad():
        for i, (image_tensor, text_ids) in enumerate(iterator):
            if i >= num_samples:
                break
            
            try:
                image_tensor = image_tensor.unsqueeze(0).to(device)
                
                # Decode ground truth
                ground_truth = ""
                for idx in text_ids:
                    if idx in idx_to_char:
                        char = idx_to_char[idx]
                        if char == '<EOS>':
                            break
                        if char not in ['<PAD>', '<BOS>', '<UNK>']:
                            ground_truth += char
                
                # Run inference
                if use_beam_search:
                    predicted_text = decode_beam_search(model, image_tensor, char_to_idx, idx_to_char, device)
                else:
                    predicted_text = decode_greedy(model, image_tensor, char_to_idx, idx_to_char, device)
                
                # Compute metrics
                cer = compute_cer(predicted_text, ground_truth)
                wer = compute_wer(predicted_text, ground_truth)
                edit_dist = levenshtein_distance(predicted_text, ground_truth)
                
                total_cer += cer
                total_wer += wer
                total_edit_distance += edit_dist
                
                # Exact match
                if predicted_text == ground_truth:
                    exact_matches += 1
                
                # Character-level accuracy
                max_len = max(len(predicted_text), len(ground_truth))
                for j in range(max_len):
                    char_total += 1
                    if j < len(predicted_text) and j < len(ground_truth):
                        if predicted_text[j] == ground_truth[j]:
                            char_correct += 1
                
                num_valid += 1
                
                # Save first 10 examples
                if len(examples) < 10:
                    examples.append({
                        'ground_truth': ground_truth,
                        'predicted': predicted_text,
                        'cer': cer,
                        'wer': wer,
                        'edit_distance': edit_dist,
                    })
                
            except Exception as e:
                if verbose:
                    print(f"\n⚠️  Error processing sample {i}: {e}")
                continue
    
    if num_valid == 0:
        raise ValueError("No valid samples processed!")
    
    # Calculate averages
    avg_cer = total_cer / num_valid
    avg_wer = total_wer / num_valid
    exact_match_rate = exact_matches / num_valid
    avg_edit_distance = total_edit_distance / num_valid
    char_accuracy = char_correct / char_total if char_total > 0 else 0.0
    
    return {
        'num_samples': num_valid,
        'cer': avg_cer,
        'wer': avg_wer,
        'exact_match_rate': exact_match_rate,
        'avg_edit_distance': avg_edit_distance,
        'char_accuracy': char_accuracy,
        'examples': examples,
    }


def print_results(metrics):
    """Pretty print evaluation results."""
    print(f"\n{'='*70}")
    print("OCR MODEL EVALUATION RESULTS")
    print(f"{'='*70}")
    
    print(f"\nSamples Evaluated: {metrics['num_samples']}")
    
    # Core metrics
    print(f"\nCORE OCR METRICS:")
    print(f"  Character Error Rate (CER): {metrics['cer']*100:.2f}%")
    print(f"    (Lower is better. Measures character-level accuracy)")
    print(f"  Word Error Rate (WER): {metrics['wer']*100:.2f}%")
    print(f"    (Lower is better. Measures word-level accuracy)")
    print(f"  Exact Match Rate: {metrics['exact_match_rate']*100:.2f}%")
    print(f"    (Perfect predictions)")
    
    print(f"\nADDITIONAL METRICS:")
    print(f"  Character Accuracy: {metrics['char_accuracy']*100:.2f}%")
    print(f"  Average Edit Distance: {metrics['avg_edit_distance']:.2f} characters")
    
    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print(f"{'='*70}")
    
    # CER assessment
    cer = metrics['cer']
    if cer < 0.02:
        cer_status = "✓ EXCELLENT - Near-perfect recognition"
    elif cer < 0.05:
        cer_status = "✓ GOOD - High accuracy"
    elif cer < 0.10:
        cer_status = "⚠ ACCEPTABLE - Usable quality"
    else:
        cer_status = "✗ POOR - Needs more training"
    
    # Exact match assessment
    exact_match = metrics['exact_match_rate']
    if exact_match > 0.80:
        match_status = "✓ EXCELLENT - Very reliable"
    elif exact_match > 0.50:
        match_status = "✓ GOOD - Mostly accurate"
    elif exact_match > 0.30:
        match_status = "⚠ ACCEPTABLE - Moderate accuracy"
    else:
        match_status = "✗ POOR - Low reliability"
    
    print(f"CER: {cer_status}")
    print(f"Exact Match: {match_status}")
    
    # Examples
    if metrics['examples']:
        print(f"\n{'='*70}")
        print("SAMPLE PREDICTIONS:")
        print(f"{'='*70}")
        
        for idx, ex in enumerate(metrics['examples']):
            print(f"\nExample {idx+1}:")
            print(f"  Ground Truth: '{ex['ground_truth']}'")
            print(f"  Predicted:    '{ex['predicted']}'")
            print(f"  CER: {ex['cer']*100:.1f}% | WER: {ex['wer']*100:.1f}% | Edit Dist: {ex['edit_distance']}")
    
    # Overall assessment
    print(f"\n{'='*70}")
    
    if cer < 0.05 and exact_match > 0.50:
        print("✓ OCR model is working excellently!")
        print("  High accuracy and reliability.")
        print("  Ready for production use.")
    elif cer < 0.10 and exact_match > 0.30:
        print("✓ OCR model is working well!")
        print("  Good baseline accuracy.")
        print("  Consider training longer for better quality.")
    else:
        print("⚠ OCR model needs more training.")
        print("  Current accuracy is below production standards.")
        print("  Recommendation: Train for more steps or increase model size.")
    
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Test OCR model with accuracy metrics")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ocr_tiny",
                       help="Path to OCR checkpoint directory")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate (default: 100)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with 10 samples")
    parser.add_argument("--beam_search", action="store_true",
                       help="Use beam search decoding (slower, better quality)")
    args = parser.parse_args()
    
    if args.quick:
        args.num_samples = 10
    
    print("=" * 70)
    print("OCR MODEL ACCURACY TEST")
    print("=" * 70)
    
    # Load model
    try:
        model, char_to_idx, idx_to_char, cfg = load_model_and_vocab(args.checkpoint, args.device)
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate
    try:
        metrics = evaluate_ocr_accuracy(
            model, char_to_idx, idx_to_char, cfg,
            device=args.device,
            num_samples=args.num_samples,
            use_beam_search=args.beam_search,
            verbose=True
        )
        
        print_results(metrics)
        
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()