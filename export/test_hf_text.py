"""
Test HuggingFace text-only model (MuOmniForCausalLM) with scored analytics.
Uses random samples from training data.

Usage:
    python -m export.test_hf_text
    python -m export.test_hf_text --num_samples 50 --device cpu
"""

import sys
import os
import argparse
import random

import torch
torch.set_float32_matmul_precision('high')

try:
    from export.modeling_muomni import MuOmniForCausalLM
except ImportError:
    # Fallback for direct script execution from export/ folder.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from modeling_muomni import MuOmniForCausalLM

# Add parent dir for omni imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omni.tokenizer import BPETokenizer
from omni.io_utils import enable_log_file, default_log_path
from omni.eval_utils import sample_lines, count_topk_from_logits, safe_perplexity


def main():
    parser = argparse.ArgumentParser(description="Test HF text model")
    parser.add_argument("--model_dir", default="export", help="Export directory")
    parser.add_argument("--corpus", default="data/text/production_corpus.txt")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_file", default=default_log_path(__file__), help="Write stdout/stderr to this file (UTF-8)")
    args = parser.parse_args()
    enable_log_file(args.log_file, header=f"test_hf_text.py start | model_dir={args.model_dir}")

    print("=" * 60)
    print("HF TEXT MODEL — SCORED TEST")
    print("=" * 60)

    model = MuOmniForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, torch_dtype=torch.float32
    ).to(args.device).eval()
    tok = BPETokenizer(os.path.join(args.model_dir, "tokenizer.model"))
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params on {args.device}")

    # Load training data
    samples = sample_lines(args.corpus, args.num_samples, min_len=11, seed=args.seed)

    # Accuracy + loss
    c1, c5, c10, total = 0, 0, 0, 0
    total_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    for line in samples:
        ids = [1] + tok.encode(line)
        if len(ids) < 4:
            continue
        x = torch.tensor([ids[:-1]], device=args.device)
        tgt = torch.tensor([ids[1:]], device=args.device)
        with torch.inference_mode():
            logits = model(input_ids=x).logits
        total_loss += loss_fn(logits[0], tgt[0]).item()
        topk = count_topk_from_logits(logits[0], tgt[0], ks=(1, 5, 10))
        c1 += topk[1]
        c5 += topk[5]
        c10 += topk[10]
        total += tgt.numel()

    avg_loss = total_loss / max(total, 1)
    ppl = safe_perplexity(avg_loss, max_exp_input=20.0)

    print(f"\nSamples: {len(samples)}, Tokens: {total}")
    print(f"Avg Loss:    {avg_loss:.4f}")
    print(f"Perplexity:  {ppl:.2f}")
    print(f"Top-1:       {c1/total*100:.2f}%")
    print(f"Top-5:       {c5/total*100:.2f}%")
    print(f"Top-10:      {c10/total*100:.2f}%")

    # Generation
    print("\nGeneration Samples:")
    prompts = ["The red cat", "Count: 1 2 3", "3 plus 4 is", "A blue circle on", "The dog likes"]
    for p in prompts:
        ids = [1] + tok.encode(p)
        x = torch.tensor([ids], device=args.device)
        with torch.inference_mode():
            gen = model.generate(x, max_new_tokens=15, do_sample=True,
                                 temperature=0.7, top_k=40, repetition_penalty=1.3)
        print(f"  {p} -> {tok.decode(gen[0].tolist())}")

    # Rating
    print("\n" + "=" * 60)
    print("RATING:")
    print(f"  Perplexity: {'EXCELLENT' if ppl < 5 else 'GOOD' if ppl < 20 else 'POOR'}")
    print(f"  Top-1:      {'EXCELLENT' if c1/total > 0.6 else 'GOOD' if c1/total > 0.3 else 'POOR'}")
    print(f"  Top-5:      {'EXCELLENT' if c5/total > 0.9 else 'GOOD' if c5/total > 0.6 else 'POOR'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
