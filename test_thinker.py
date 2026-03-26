"""
Complete test script to validate Thinker language model accuracy.
Measures perplexity, next-token accuracy, generation quality, and coherence.
"""

import torch
from torch.nn import CrossEntropyLoss
import json
import os
import argparse
import random
import numpy as np
from omni.thinker import ThinkerLM
from omni.tokenizer import BPETokenizer
from omni.utils import find_checkpoint, strip_orig_mod
from tqdm import tqdm

torch.set_float32_matmul_precision('high')


def load_model_and_tokenizer(checkpoint_dir, device="cuda"):
    """Load Thinker model and tokenizer from checkpoint."""
    checkpoint_path, checkpoint = find_checkpoint(checkpoint_dir, "thinker.pt", "thinker_step_", device)
    if checkpoint is None:
        raise FileNotFoundError(f"Checkpoint not found in: {checkpoint_dir}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Config must be in checkpoint dir (saved during training)
    if "config" in checkpoint:
        cfg = checkpoint["config"]
    else:
        config_path = os.path.join(checkpoint_dir, "config.json")
        if os.path.exists(config_path):
            print(f"Loading config from: {config_path}")
            with open(config_path, 'r') as f:
                cfg = json.load(f)
        else:
            raise FileNotFoundError(f"Config not found: {config_path}. Re-run training to generate it.")
    
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
        use_ltc=cfg.get("use_ltc", False),
        window_size=cfg.get("window_size", 0)
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
    Compute perplexity and in-distribution next-token prediction accuracy.

    Loads random lines from the training corpus, tokenizes each line, feeds the
    token sequence to the model, and checks whether the predicted next token
    matches the actual next token at every position.

    Metrics:
    - Perplexity: exp(cross-entropy loss). Lower is better.
    - Top-1 accuracy: Exact next-token prediction
    - Top-5 accuracy: Next token in top 5 predictions
    - Top-10 accuracy: Next token in top 10 predictions
    """
    model.eval()
    loss_fn = CrossEntropyLoss(reduction='none')

    # Load random lines from the training corpus
    text_path = cfg.get("train_text", "data/text/production_corpus.txt")
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Text file not found: {text_path}")

    with open(text_path, 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f if len(line.strip()) >= 10]

    if not all_lines:
        raise ValueError("Training corpus is empty or has no lines with >= 10 characters")

    random.seed(42)
    sampled_lines = random.sample(all_lines, min(num_samples, len(all_lines)))

    ctx_len = cfg.get("ctx_len", 512)

    # Accumulators
    total_loss = 0.0
    total_tokens = 0
    correct_top1 = 0
    correct_top5 = 0
    correct_top10 = 0
    total_predictions = 0

    if verbose:
        print(f"\nEvaluating in-distribution accuracy on {len(sampled_lines)} corpus lines...")
        iterator = tqdm(enumerate(sampled_lines), total=len(sampled_lines), desc="Processing")
    else:
        iterator = enumerate(sampled_lines)

    with torch.inference_mode():
        for i, line in iterator:
            try:
                # Tokenize the line: BOS + encoded tokens
                token_ids = [1] + tokenizer.encode(line)
                if len(token_ids) < 3:
                    continue  # Skip lines that are too short

                # Truncate to context length
                if len(token_ids) > ctx_len:
                    token_ids = token_ids[:ctx_len]

                # Input is all tokens except the last; target is all tokens except the first
                x = torch.tensor([token_ids[:-1]], device=device)  # (1, T)
                y = torch.tensor([token_ids[1:]], device=device)   # (1, T)

                # Forward pass
                logits = model(x)  # (1, T, vocab_size)

                seq_len = y.size(1)

                # Compute per-token loss
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))  # (T,)
                total_loss += loss.sum().item()
                total_tokens += seq_len

                # Top-1 accuracy
                preds_top1 = logits.argmax(dim=-1)  # (1, T)
                correct_top1 += (preds_top1 == y).sum().item()

                # Top-5 accuracy
                preds_top5 = logits.topk(5, dim=-1).indices  # (1, T, 5)
                y_expanded5 = y.unsqueeze(-1).expand(-1, -1, 5)  # (1, T, 5)
                correct_top5 += (preds_top5 == y_expanded5).any(dim=-1).sum().item()

                # Top-10 accuracy
                preds_top10 = logits.topk(10, dim=-1).indices  # (1, T, 10)
                y_expanded10 = y.unsqueeze(-1).expand(-1, -1, 10)  # (1, T, 10)
                correct_top10 += (preds_top10 == y_expanded10).any(dim=-1).sum().item()

                total_predictions += seq_len

            except Exception as e:
                if verbose:
                    print(f"\n  Warning: Error processing line {i}: {e}")
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
        'num_samples': len(sampled_lines),
        'total_tokens': total_tokens,
        'avg_loss': avg_loss,
        'perplexity': perplexity,
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'top10_accuracy': top10_accuracy,
    }


def load_corpus_lines(cfg, min_length=10):
    """Load non-empty lines from the training corpus, filtered by minimum character length."""
    text_path = cfg.get("train_text", "data/text/production_corpus.txt")
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Training corpus not found: {text_path}")
    with open(text_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if len(line.strip()) >= min_length]
    return lines


def make_prompt_from_line(line, tokenizer, prompt_token_count=4):
    """
    Split a corpus line into a prompt (first N tokens) and the expected continuation.

    Returns (prompt_text, full_line) or None if the line is too short.
    """
    token_ids = tokenizer.encode(line)
    if len(token_ids) < prompt_token_count + 2:
        return None  # Line too short to split meaningfully
    prompt_ids = token_ids[:prompt_token_count]
    prompt_text = tokenizer.decode(prompt_ids)
    return prompt_text, line


def evaluate_generation_quality(model, tokenizer, cfg, device="cuda", num_prompts=10):
    """
    Evaluate generation quality using in-distribution prompts from the training corpus.

    Reads random lines from the training data, uses the first few tokens of each line
    as the prompt, and generates continuations via model.generate().

    Returns generated texts and basic quality metrics.
    """
    model.eval()

    # Load prompts from the actual training corpus
    corpus_lines = load_corpus_lines(cfg)
    if not corpus_lines:
        raise ValueError("Training corpus is empty or not found")

    random.seed(42)
    sampled_lines = random.sample(corpus_lines, min(num_prompts * 2, len(corpus_lines)))

    generated_samples = []

    print(f"\nGenerating {num_prompts} text samples from training corpus prompts...")

    for line in sampled_lines:
        if len(generated_samples) >= num_prompts:
            break

        result = make_prompt_from_line(line, tokenizer, prompt_token_count=4)
        if result is None:
            continue
        prompt, full_line = result

        try:
            # Encode the prompt and prepend BOS token
            prompt_ids = [1] + tokenizer.encode(prompt)
            x = torch.tensor([prompt_ids], device=device)

            # Use model.generate() for sampling
            with torch.inference_mode():
                output_ids = model.generate(
                    x,
                    max_new_tokens=50,
                    temperature=0.7,
                    top_k=40,
                    top_p=0.9,
                    repetition_penalty=1.3,
                )

            # Decode the full output (prompt + generated tokens)
            generated_text = tokenizer.decode(output_ids[0].tolist())

            generated_samples.append({
                'prompt': prompt,
                'expected': full_line,
                'generated': generated_text,
            })

        except Exception as e:
            print(f"  Error generating text for prompt '{prompt}': {e}")
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
            print(f"  Prompt:    '{sample['prompt']}'")
            if 'expected' in sample:
                print(f"  Expected:  '{sample['expected']}'")
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