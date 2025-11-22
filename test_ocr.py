"""
Simple test script to verify OCR model is working properly.
Loads a checkpoint, processes an image, and displays the extracted text.
"""

import torch
import json
import os
import argparse
import random
from PIL import Image
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from omni.ocr_model import OCRModel
from omni.utils import OCRDataset, find_checkpoint, strip_orig_mod

def load_model(checkpoint_dir, device="cuda"):
    """Load OCR model from checkpoint."""
    checkpoint_path, checkpoint = find_checkpoint(checkpoint_dir, "ocr.pt", "ocr_step_", device)
    if checkpoint is None:
        raise FileNotFoundError(f"Checkpoint not found in: {checkpoint_dir}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Get config from checkpoint or use defaults
    if "config" in checkpoint:
        cfg = checkpoint["config"]
    else:
        # Default config matching ocr_tiny.json
        cfg = {
            "img_size": 224,
            "patch": 16,
            "vision_d_model": 128,
            "vision_layers": 4,
            "vision_heads": 2,
            "vision_d_ff": 512,
            "decoder_d_model": 256,
            "decoder_layers": 4,
            "decoder_heads": 4,
            "decoder_d_ff": 1024,
            "dropout": 0.1,
            "use_gqa": False,
            "use_swiglu": True,
            "use_flash": True,
        }
    
    # Get vocabulary
    if "char_to_idx" in checkpoint and "idx_to_char" in checkpoint:
        char_to_idx = checkpoint["char_to_idx"]
        idx_to_char = checkpoint["idx_to_char"]
        vocab_size = len(char_to_idx)
        print(f"Vocabulary size: {vocab_size}")
    else:
        raise ValueError("Checkpoint missing char_to_idx and idx_to_char. Cannot decode text.")
    
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
        compile_model=False  # Disable compilation for testing
    ).to(device)
    
    # Load weights (strip _orig_mod prefixes if present from torch.compile)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Strip _orig_mod (matches training script behavior)
    state_dict = strip_orig_mod(state_dict)
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    print("✓ Model loaded successfully")
    
    return model, idx_to_char, char_to_idx, cfg

def preprocess_image(image_path, img_size=224):
    """Load and preprocess image for OCR."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor, img

def decode_text(logits, idx_to_char, max_length=256):
    """Decode logits to text string."""
    # Get predicted token IDs (greedy decoding)
    pred_ids = torch.argmax(logits, dim=-1)  # (B, T)
    
    # Convert to list
    pred_ids = pred_ids[0].cpu().tolist()  # First batch item
    
    # Decode to text
    text = []
    for idx in pred_ids:
        if idx in idx_to_char:
            char = idx_to_char[idx]
            # Stop at EOS token
            if char == '<EOS>':
                break
            # Skip special tokens
            if char not in ['<PAD>', '<BOS>', '<UNK>']:
                text.append(char)
    
    return ''.join(text)

def test_ocr_inference(model, image_tensor, idx_to_char, char_to_idx, device="cuda", max_length=256):
    """Run OCR inference on an image."""
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # Start with BOS token
        bos_id = char_to_idx.get('<BOS>', 1)
        current_ids = torch.tensor([[bos_id]], device=device)  # (B=1, T=1)
        
        # Autoregressive generation
        generated_text = []
        for step in range(max_length):
            # Forward pass
            logits = model(image_tensor, current_ids)  # (B, T, vocab_size)
            
            # Get next token (greedy)
            next_token_logits = logits[0, -1, :]  # Last position logits
            next_token_id = torch.argmax(next_token_logits).item()
            
            # Check for EOS
            if next_token_id in idx_to_char:
                char = idx_to_char[next_token_id]
                if char == '<EOS>':
                    break
                if char not in ['<PAD>', '<BOS>', '<UNK>']:
                    generated_text.append(char)
            
            # Append to sequence for next iteration
            current_ids = torch.cat([current_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
        
        return ''.join(generated_text)

def get_random_image_from_dataset(cfg, char_to_idx, idx_to_char):
    """Get a random image from the OCR dataset."""
    csv_path = cfg.get("train_csv", "data/ocr/production_ocr.csv")
    image_root = cfg.get("image_root", "data/ocr")
    img_size = cfg.get("img_size", 224)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"OCR CSV not found: {csv_path}")
    
    # Create dataset with shuffling to get random sample
    dataset = OCRDataset(
        csv_path=csv_path,
        image_root=image_root,
        img_size=img_size,
        cfg=cfg,
        shuffle_buffer_size=10000,  # Enable shuffling for randomness
        seed=random.randint(0, 1000000),  # Random seed for different samples each time
        skip_samples=0
    )
    
    # Use the dataset's vocabulary (should match checkpoint)
    dataset.char_to_idx = char_to_idx
    dataset.idx_to_char = idx_to_char
    
    # Get first sample from iterator (will be random due to shuffling)
    try:
        iterator = iter(dataset)
        image_tensor, text_ids = next(iterator)
        
        # Decode ground truth text for comparison
        ground_truth = ""
        for idx in text_ids:
            if idx in idx_to_char:
                char = idx_to_char[idx]
                if char == '<EOS>':
                    break
                if char not in ['<PAD>', '<BOS>', '<UNK>']:
                    ground_truth += char
        
        return image_tensor.unsqueeze(0), ground_truth  # Add batch dimension
    except StopIteration:
        raise ValueError("Dataset is empty - no samples found")
    except Exception as e:
        raise RuntimeError(f"Error getting sample from dataset: {e}")

def calculate_edit_distance(str1, str2):
    """Calculate Levenshtein edit distance between two strings."""
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]

def evaluate_ocr_model(model, idx_to_char, char_to_idx, cfg, device="cuda", num_samples=100):
    """Evaluate OCR model on multiple samples and compute metrics."""
    model.eval()
    loss_fn = CrossEntropyLoss(ignore_index=0)
    
    csv_path = cfg.get("train_csv", "data/ocr/production_ocr.csv")
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
        seed=random.randint(0, 1000000),
        skip_samples=0
    )
    
    dataset.char_to_idx = char_to_idx
    dataset.idx_to_char = idx_to_char
    
    iterator = iter(dataset)
    total_loss = 0.0
    total_chars = 0
    exact_matches = 0
    total_edit_distance = 0
    char_correct = 0
    char_total = 0
    
    print(f"\nEvaluating on {num_samples} samples...")
    
    with torch.no_grad():
        for i in range(num_samples):
            try:
                image_tensor, text_ids = next(iterator)
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
                extracted_text = test_ocr_inference(model, image_tensor, idx_to_char, char_to_idx, device)
                
                # Calculate edit distance
                edit_dist = calculate_edit_distance(extracted_text, ground_truth)
                total_edit_distance += edit_dist
                
                # Exact match
                if extracted_text == ground_truth:
                    exact_matches += 1
                
                # Character-level accuracy
                max_len = max(len(extracted_text), len(ground_truth))
                for j in range(max_len):
                    char_total += 1
                    if j < len(extracted_text) and j < len(ground_truth):
                        if extracted_text[j] == ground_truth[j]:
                            char_correct += 1
                
                # Compute loss (teacher forcing)
                bos_id = char_to_idx.get('<BOS>', 1)
                input_ids = torch.tensor([[bos_id]], device=device)
                total_loss_batch = 0.0
                num_steps = 0
                
                for step, target_id in enumerate(text_ids[:min(len(text_ids), 128)]):
                    if target_id == char_to_idx.get('<EOS>', 2):
                        break
                    logits = model(image_tensor, input_ids)
                    target_tensor = torch.tensor([[target_id]], device=device)
                    step_loss = loss_fn(logits[:, -1:, :].view(-1, logits.size(-1)), target_tensor.view(-1))
                    total_loss_batch += step_loss.item()
                    num_steps += 1
                    input_ids = torch.cat([input_ids, target_tensor], dim=1)
                
                if num_steps > 0:
                    total_loss += total_loss_batch / num_steps
                    total_chars += num_steps
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{num_samples} samples...", end='\r')
                    
            except StopIteration:
                print(f"\n⚠️  Dataset exhausted after {i} samples")
                break
            except Exception as e:
                print(f"\n⚠️  Error processing sample {i}: {e}")
                continue
    
    print()  # New line after progress
    
    # Calculate metrics
    avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
    exact_match_rate = (exact_matches / num_samples * 100) if num_samples > 0 else 0.0
    avg_edit_distance = total_edit_distance / num_samples if num_samples > 0 else 0.0
    char_accuracy = (char_correct / char_total * 100) if char_total > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'exact_match_rate': exact_match_rate,
        'char_accuracy': char_accuracy,
        'avg_edit_distance': avg_edit_distance,
        'num_samples': i + 1
    }

def main():
    parser = argparse.ArgumentParser(description="Test OCR model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ocr_tiny",
                       help="Path to OCR checkpoint directory")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate (default: 100)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda/cpu)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("OCR Model Test")
    print("=" * 60)
    
    # Load model
    try:
        model, idx_to_char, char_to_idx, cfg = load_model(args.checkpoint, args.device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate on multiple samples from dataset
    try:
        metrics = evaluate_ocr_model(model, idx_to_char, char_to_idx, cfg, args.device, args.num_samples)
        
        print(f"\n{'=' * 60}")
        print("EVALUATION RESULTS:")
        print(f"{'=' * 60}")
        print(f"Samples evaluated: {metrics['num_samples']}")
        print(f"Average Loss: {metrics['loss']:.4f}")
        print(f"Exact Match Rate: {metrics['exact_match_rate']:.2f}%")
        print(f"Character Accuracy: {metrics['char_accuracy']:.2f}%")
        print(f"Average Edit Distance: {metrics['avg_edit_distance']:.2f}")
        print(f"{'=' * 60}")
        
        # Interpretation
        if metrics['exact_match_rate'] > 80:
            print("✓ Excellent OCR performance!")
        elif metrics['exact_match_rate'] > 50:
            print("✓ Good OCR performance")
        elif metrics['exact_match_rate'] > 20:
            print("⚠️  Moderate performance - model may need more training")
        else:
            print("⚠️  Poor performance - model needs significant training")
            
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()

