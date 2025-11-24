"""
Simple test script to verify Thinker (language model) is working properly.
Loads a checkpoint, generates text from a prompt, and displays the output.
"""

import torch
import json
import os
import argparse
import random
from omni.thinker import ThinkerLM
from omni.tokenizer import BPETokenizer
from omni.utils import TextDataset, find_checkpoint, strip_orig_mod
from torch.nn import CrossEntropyLoss


def load_model(checkpoint_dir, device="cuda"):
    """Load Thinker model from checkpoint."""
    checkpoint_path, checkpoint = find_checkpoint(checkpoint_dir, "thinker.pt", "thinker_step_", device)
    if checkpoint is None:
        raise FileNotFoundError(f"Checkpoint not found in: {checkpoint_dir}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Get config from checkpoint or use defaults
    if "config" in checkpoint:
        cfg = checkpoint["config"]
    else:
        cfg = {
            "vocab_size": 32000,
            "n_layers": 4,
            "d_model": 256,
            "n_heads": 4,
            "d_ff": 1024,
            "dropout": 0.1,
            "rope_theta": 10000,
            "ctx_len": 512,
            "use_gqa": False,
            "use_swiglu": True,
        }
    
    # Load tokenizer
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.model")
    if not os.path.exists(tokenizer_path):
        # Try default location
        tokenizer_path = "checkpoints/thinker_tiny/tokenizer.model"
    
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    tokenizer = BPETokenizer(tokenizer_path)
    vocab_size = tokenizer.sp.get_piece_size()
    print(f"Tokenizer vocabulary size: {vocab_size}")
    
    # Initialize model
    model = ThinkerLM(
        vocab=vocab_size,
        n_layers=cfg.get("n_layers", 4),
        d=cfg.get("d_model", 256),
        heads=cfg.get("n_heads", 4),
        ff=cfg.get("d_ff", 1024),
        dropout=cfg.get("dropout", 0.1),
        rope_theta=cfg.get("rope_theta", 10000),
        ctx=cfg.get("ctx_len", 512),
        use_gqa=cfg.get("use_gqa", False),
        use_swiglu=cfg.get("use_swiglu", True),
        use_moe=cfg.get("use_moe", False),
        num_experts=cfg.get("num_experts", 8),
        num_experts_per_tok=cfg.get("num_experts_per_tok", 2),
        compile_model=False
    ).to(device)
    
    # Load weights
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Normalize state dict (only strip _orig_mod, don't convert attention weights)
    # This matches training script behavior - load checkpoints as saved
    from omni.utils import strip_orig_mod
    state_dict = strip_orig_mod(state_dict)
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    print("✓ Model loaded successfully")
    
    return model, tokenizer, cfg

def generate_text(model, tokenizer, prompt, device="cuda", max_length=100, temperature=0.8, ctx_len=512):
    """Generate text from a prompt."""
    model.eval()
    
    # Encode prompt
    prompt_ids = [1] + tokenizer.encode(prompt)  # Add BOS token
    prompt_tensor = torch.tensor([prompt_ids], device=device)
    
    generated_ids = prompt_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            logits = model(prompt_tensor)  # (B, T, vocab_size)
            
            # Get next token (with temperature)
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()
            
            # Stop at EOS or if sequence too long
            if next_token_id == 0 or len(generated_ids) >= ctx_len - 1:
                break
            
            generated_ids.append(next_token_id)
            
            # Update input for next iteration
            prompt_tensor = torch.tensor([[next_token_id]], device=device)
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids)
    return generated_text

def get_random_text_from_dataset(cfg, tokenizer):
    """Get a random text sample from the dataset."""
    text_path = cfg.get("train_text", "data/text/production_corpus.txt")
    
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Text file not found: {text_path}")
    
    # Create dataset with shuffling
    dataset = TextDataset(
        path=text_path,
        tokenizer=tokenizer,
        ctx=cfg.get("ctx_len", 512),
        shuffle_buffer_size=10000,
        seed=random.randint(0, 1000000),
        skip_samples=0
    )
    
    # Get first sample
    try:
        iterator = iter(dataset)
        x, y = next(iterator)
        # Decode the text
        text = tokenizer.decode(x.tolist())
        return text.strip()
    except StopIteration:
        raise ValueError("Dataset is empty")

def evaluate_model(model, tokenizer, cfg, device="cuda", num_samples=100):
    """Evaluate model on multiple samples and compute metrics."""
    model.eval()
    loss_fn = CrossEntropyLoss(ignore_index=0)
    
    text_path = cfg.get("train_text", "data/text/production_corpus.txt")
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Text file not found: {text_path}")
    
    # Create dataset
    dataset = TextDataset(
        path=text_path,
        tokenizer=tokenizer,
        ctx=cfg.get("ctx_len", 512),
        shuffle_buffer_size=10000,
        seed=random.randint(0, 1000000),
        skip_samples=0
    )
    
    iterator = iter(dataset)
    total_loss = 0.0
    total_tokens = 0
    correct_predictions = 0
    total_predictions = 0
    
    print(f"\nEvaluating on {num_samples} samples...")
    
    with torch.no_grad():
        for i in range(num_samples):
            try:
                x, y = next(iterator)
                x = x.unsqueeze(0).to(device)  # Add batch dimension
                y = y.unsqueeze(0).to(device)
                
                # Forward pass
                logits = model(x)  # (B, T, vocab_size)
                
                # Compute loss
                logits_flat = logits.view(-1, logits.size(-1))
                y_flat = y.view(-1)
                loss = loss_fn(logits_flat, y_flat)
                
                # Compute accuracy (next token prediction)
                preds = torch.argmax(logits[:, :-1, :], dim=-1)  # Predictions for positions 0 to T-1
                targets = y[:, 1:]  # Targets for positions 1 to T
                mask = (targets != 0)  # Ignore padding
                
                correct = (preds == targets) & mask
                correct_predictions += correct.sum().item()
                total_predictions += mask.sum().item()
                
                # Accumulate loss
                num_valid_tokens = (y_flat != 0).sum().item()
                if num_valid_tokens > 0:
                    total_loss += loss.item() * num_valid_tokens
                    total_tokens += num_valid_tokens
                
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
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0.0
    perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss < 10 else float('inf')
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'accuracy': accuracy,
        'num_samples': i + 1,
        'total_tokens': total_tokens
    }

def main():
    parser = argparse.ArgumentParser(description="Test Thinker language model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/thinker_tiny",
                       help="Path to Thinker checkpoint directory")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate (default: 100)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda/cpu)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Thinker Language Model Test")
    print("=" * 60)
    
    # Load model
    try:
        model, tokenizer, cfg = load_model(args.checkpoint, args.device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate on multiple samples from dataset
    try:
        metrics = evaluate_model(model, tokenizer, cfg, args.device, args.num_samples)
        
        print(f"\n{'=' * 60}")
        print("EVALUATION RESULTS:")
        print(f"{'=' * 60}")
        print(f"Samples evaluated: {metrics['num_samples']}")
        print(f"Total tokens: {metrics['total_tokens']:,}")
        print(f"Average Loss: {metrics['loss']:.4f}")
        print(f"Perplexity: {metrics['perplexity']:.2f}")
        print(f"Accuracy (next token): {metrics['accuracy']:.2f}%")
        print(f"{'=' * 60}")
        
        # Interpretation
        if metrics['perplexity'] < 10:
            print("✓ Excellent model performance!")
        elif metrics['perplexity'] < 50:
            print("✓ Good model performance")
        elif metrics['perplexity'] < 100:
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

