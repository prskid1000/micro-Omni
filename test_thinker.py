"""
Complete test script to validate Thinker language model accuracy.
Measures perplexity, next-token accuracy, generation quality, and coherence.
"""

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import json
import os
import argparse
import random
import numpy as np
from omni.thinker import ThinkerLM
from omni.tokenizer import BPETokenizer
from omni.utils import TextDataset, find_checkpoint, strip_orig_mod
from tqdm import tqdm

torch.set_float32_matmul_precision('high')


def load_model_and_tokenizer(checkpoint_dir, device="cuda"):
    """Load Thinker model and tokenizer from checkpoint."""
    checkpoint_path, checkpoint = find_checkpoint(checkpoint_dir, "thinker.pt", "thinker_step_", device)
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
    
    # Load tokenizer
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.model")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    tokenizer = BPETokenizer(tokenizer_path)
    vocab_size = tokenizer.sp.get_piece_size()
    
    print(f"✓ Loaded tokenizer (vocab size: {vocab_size})")
    
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
        compile_model=False,
        use_spiking=cfg.get("use_spiking", False),
        use_ltc=cfg.get("use_ltc", False)
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
    print(f"  Layers: {cfg.get('n_layers', 4)}")
    print(f"  Model dimension: {cfg.get('d_model', 256)}")
    print(f"  Context length: {cfg.get('ctx_len', 512)}")
    
    return model, tokenizer, cfg


def compute_perplexity_and_accuracy(model, tokenizer, cfg, device="cuda", num_samples=100, verbose=True):
    """
    Compute perplexity and next-token prediction accuracy.
    
    Metrics:
    - Perplexity: exp(cross-entropy loss). Lower is better.
    - Top-1 accuracy: Exact next-token prediction
    - Top-5 accuracy: Next token in top 5 predictions
    - Top-10 accuracy: Next token in top 10 predictions
    """
    model.eval()
    loss_fn = CrossEntropyLoss(ignore_index=0, reduction='none')
    
    text_path = cfg.get("train_text", "data/text/production_corpus.txt")
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Text file not found: {text_path}")
    
    # Create dataset
    dataset = TextDataset(
        path=text_path,
        tokenizer=tokenizer,
        ctx=cfg.get("ctx_len", 512),
        shuffle_buffer_size=10000,
        seed=42,  # Fixed seed for reproducibility
        skip_samples=0
    )
    
    # Accumulators
    total_loss = 0.0
    total_tokens = 0
    correct_top1 = 0
    correct_top5 = 0
    correct_top10 = 0
    total_predictions = 0
    
    if verbose:
        print(f"\nEvaluating perplexity on {num_samples} samples...")
        iterator = tqdm(dataset, total=num_samples, desc="Processing")
    else:
        iterator = iter(dataset)
    
    with torch.inference_mode():
        for i, (x, y) in enumerate(iterator):
            if i >= num_samples:
                break
            
            try:
                x = x.unsqueeze(0).to(device)  # (1, T)
                y = y.unsqueeze(0).to(device)  # (1, T)
                
                # Forward pass
                logits = model(x)  # (1, T, vocab_size)
                
                # Compute per-token loss (ignoring padding)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))  # (T,)
                mask = (y.view(-1) != 0).float()  # Mask for non-padding tokens
                
                # Accumulate loss for perplexity
                valid_tokens = mask.sum().item()
                if valid_tokens > 0:
                    total_loss += (loss * mask).sum().item()
                    total_tokens += valid_tokens
                
                # Compute accuracy for next-token prediction
                # Predictions for positions 0 to T-1, targets for positions 1 to T
                # But we only evaluate on valid (non-padding) targets
                preds_logits = logits[:, :-1, :]  # (1, T-1, vocab)
                targets = y[:, 1:]  # (1, T-1)
                target_mask = (targets != 0)  # (1, T-1)
                
                # Top-1 accuracy
                preds_top1 = preds_logits.argmax(dim=-1)  # (1, T-1)
                correct_top1 += ((preds_top1 == targets) & target_mask).sum().item()
                
                # Top-5 accuracy
                preds_top5 = preds_logits.topk(5, dim=-1).indices  # (1, T-1, 5)
                targets_expanded = targets.unsqueeze(-1).expand(-1, -1, 5)  # (1, T-1, 5)
                correct_top5 += ((preds_top5 == targets_expanded).any(dim=-1) & target_mask).sum().item()
                
                # Top-10 accuracy
                preds_top10 = preds_logits.topk(10, dim=-1).indices  # (1, T-1, 10)
                targets_expanded_10 = targets.unsqueeze(-1).expand(-1, -1, 10)  # (1, T-1, 10)
                correct_top10 += ((preds_top10 == targets_expanded_10).any(dim=-1) & target_mask).sum().item()
                
                total_predictions += target_mask.sum().item()
                
            except Exception as e:
                if verbose:
                    print(f"\n⚠️  Error processing sample {i}: {e}")
                continue
    
    if total_tokens == 0 or total_predictions == 0:
        raise ValueError("No valid tokens processed!")
    
    # Calculate metrics
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss) if avg_loss < 10 else float('inf')
    top1_accuracy = correct_top1 / total_predictions
    top5_accuracy = correct_top5 / total_predictions
    top10_accuracy = correct_top10 / total_predictions
    
    return {
        'num_samples': i + 1,
        'total_tokens': total_tokens,
        'avg_loss': avg_loss,
        'perplexity': perplexity,
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'top10_accuracy': top10_accuracy,
    }


def generate_text(model, tokenizer, prompt, device="cuda", max_length=100, temperature=0.8, top_k=50, top_p=0.95, ctx_len=512):
    """
    Generate text from a prompt with advanced sampling.
    
    Args:
        model: ThinkerLM model
        tokenizer: BPETokenizer
        prompt: Input text prompt
        device: Device to run on
        max_length: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top k tokens for sampling
        top_p: Nucleus sampling threshold
        ctx_len: Context length limit
    
    Returns:
        Generated text
    """
    model.eval()
    
    # Encode prompt
    prompt_ids = [1] + tokenizer.encode(prompt)  # Add BOS token
    if len(prompt_ids) > ctx_len - max_length:
        prompt_ids = prompt_ids[-(ctx_len - max_length):]
    
    generated_ids = prompt_ids.copy()
    
    with torch.inference_mode():
        for _ in range(max_length):
            # Get context window (last ctx_len tokens)
            context = generated_ids[-ctx_len:]
            input_tensor = torch.tensor([context], device=device)
            
            # Forward pass
            logits = model(input_tensor)  # (1, T, vocab_size)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits[top_k_indices] = top_k_logits
            
            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()
            
            # Stop at EOS (token 0) or special tokens
            if next_token_id == 0:
                break
            
            generated_ids.append(next_token_id)
            
            # Stop if exceeding context length
            if len(generated_ids) >= ctx_len:
                break
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids)
    return generated_text


def evaluate_generation_quality(model, tokenizer, cfg, device="cuda", num_prompts=10):
    """
    Evaluate generation quality with sample prompts.
    
    Returns generated texts and basic quality metrics.
    """
    model.eval()
    
    # Sample prompts (you can customize these)
    prompts = [
        "The quick brown fox",
        "Once upon a time",
        "In a world where",
        "The meaning of life is",
        "Artificial intelligence will",
        "The future of technology",
        "Scientists have discovered",
        "The most important thing",
        "When I was young",
        "The best way to learn",
    ]
    
    generated_samples = []
    
    print(f"\nGenerating {num_prompts} text samples...")
    
    for i, prompt in enumerate(prompts[:num_prompts]):
        try:
            generated_text = generate_text(
                model, tokenizer, prompt,
                device=device,
                max_length=50,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                ctx_len=cfg.get("ctx_len", 512)
            )
            
            generated_samples.append({
                'prompt': prompt,
                'generated': generated_text,
            })
            
        except Exception as e:
            print(f"⚠️  Error generating text for prompt '{prompt}': {e}")
            continue
    
    return generated_samples


def print_results(ppl_metrics, generation_samples=None):
    """Pretty print evaluation results."""
    print(f"\n{'='*70}")
    print("THINKER LANGUAGE MODEL EVALUATION RESULTS")
    print(f"{'='*70}")
    
    print(f"\nSamples Evaluated: {ppl_metrics['num_samples']}")
    print(f"Total Tokens: {ppl_metrics['total_tokens']:,}")
    
    # Core metrics
    print(f"\nCORE LANGUAGE MODELING METRICS:")
    print(f"  Average Loss: {ppl_metrics['avg_loss']:.4f}")
    print(f"  Perplexity: {ppl_metrics['perplexity']:.2f}")
    print(f"    (Lower is better. Random baseline ≈ vocab_size)")
    
    print(f"\nNEXT-TOKEN PREDICTION ACCURACY:")
    print(f"  Top-1 Accuracy: {ppl_metrics['top1_accuracy']*100:.2f}%")
    print(f"  Top-5 Accuracy: {ppl_metrics['top5_accuracy']*100:.2f}%")
    print(f"  Top-10 Accuracy: {ppl_metrics['top10_accuracy']*100:.2f}%")
    
    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print(f"{'='*70}")
    
    # Perplexity assessment
    ppl = ppl_metrics['perplexity']
    if ppl < 10:
        ppl_status = "✓ EXCELLENT - Strong language understanding"
    elif ppl < 30:
        ppl_status = "✓ GOOD - Solid performance"
    elif ppl < 100:
        ppl_status = "⚠ ACCEPTABLE - Reasonable but could improve"
    else:
        ppl_status = "✗ POOR - Needs more training"
    
    # Accuracy assessment
    top1_acc = ppl_metrics['top1_accuracy']
    if top1_acc > 0.50:
        acc_status = "✓ EXCELLENT - Very accurate predictions"
    elif top1_acc > 0.35:
        acc_status = "✓ GOOD - Good prediction accuracy"
    elif top1_acc > 0.20:
        acc_status = "⚠ ACCEPTABLE - Learning patterns"
    else:
        acc_status = "✗ POOR - Needs significant training"
    
    print(f"Perplexity: {ppl_status}")
    print(f"Prediction Accuracy: {acc_status}")
    
    # Generation samples
    if generation_samples:
        print(f"\n{'='*70}")
        print("SAMPLE GENERATIONS:")
        print(f"{'='*70}")
        
        for idx, sample in enumerate(generation_samples):
            print(f"\nSample {idx+1}:")
            print(f"  Prompt: '{sample['prompt']}'")
            print(f"  Generated: '{sample['generated']}'")
    
    # Overall assessment
    print(f"\n{'='*70}")
    
    if ppl < 30 and top1_acc > 0.35:
        print("✓ Thinker is working excellently!")
        print("  Strong language modeling capabilities.")
        print("  Ready for downstream tasks (classification, generation, etc.)")
    elif ppl < 100 and top1_acc > 0.20:
        print("✓ Thinker is working well!")
        print("  Good baseline for language understanding.")
        print("  Consider training longer for better quality.")
    else:
        print("⚠ Thinker needs more training.")
        print("  Current quality is below production standards.")
        print("  Recommendation: Train for more steps or check hyperparameters.")
    
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Test Thinker language model with accuracy metrics")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/thinker_tiny",
                       help="Path to Thinker checkpoint directory")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate (default: 100)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with 10 samples")
    parser.add_argument("--generate", action="store_true",
                       help="Also generate sample texts")
    args = parser.parse_args()
    
    if args.quick:
        args.num_samples = 10
    
    print("=" * 70)
    print("THINKER LANGUAGE MODEL ACCURACY TEST")
    print("=" * 70)

    if args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Load model
    try:
        model, tokenizer, cfg = load_model_and_tokenizer(args.checkpoint, args.device)
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate perplexity and accuracy
    try:
        ppl_metrics = compute_perplexity_and_accuracy(
            model, tokenizer, cfg,
            device=args.device,
            num_samples=args.num_samples,
            verbose=True
        )
        
        # Optionally generate text samples
        generation_samples = None
        if args.generate:
            generation_samples = evaluate_generation_quality(
                model, tokenizer, cfg,
                device=args.device,
                num_prompts=5
            )
        
        print_results(ppl_metrics, generation_samples)
        
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()