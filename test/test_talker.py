"""
Complete test script to validate Talker (TTS) accuracy.
Measures codec quality, autoregressive prediction accuracy, and mel reconstruction.
"""

import torch
import torch.nn as nn
import json
import os
import argparse
import random
import numpy as np
from omni.talker import TalkerTiny
from omni.codec import RVQ
from omni.data_utils import TTSDataset
from omni.checkpoint_utils import find_checkpoint, strip_orig_mod
from omni.io_utils import enable_log_file, default_log_path
from omni.eval_utils import load_checkpoint_and_config
from omni.training_utils import MetricsLogger, build_run_id
from tqdm import tqdm

torch.set_float32_matmul_precision('high')


def load_model_and_codec(checkpoint_dir, device="cuda", config_path=None):
    """Load Talker model and RVQ codec from checkpoint."""
    _, checkpoint, cfg = load_checkpoint_and_config(
        checkpoint_dir, "talker.pt", "talker_step_", device=device, config_path=config_path
    )

    
    codebooks = cfg.get("codebooks", 2)
    codebook_size = cfg.get("codebook_size", 128)
    
    # Initialize models
    rvq = RVQ(
        codebooks=codebooks,
        codebook_size=codebook_size,
        d=64,
        compile_model=False
    ).to(device)
    
    talker = TalkerTiny(
        d=cfg.get("d_model", cfg.get("d", 384)),
        n_layers=cfg.get("n_layers", 4),
        n_heads=cfg.get("n_heads", 4),
        ff=cfg.get("d_ff", cfg.get("ff", 1536)),
        codebooks=codebooks,
        codebook_size=codebook_size,
        dropout=cfg.get("dropout", 0.1),
        use_gqa=cfg.get("use_gqa", False),
        use_swiglu=cfg.get("use_swiglu", True),
        rope_theta=cfg.get("rope_theta", 10000.0),
        compile_model=False,
        use_spiking=cfg.get("use_spiking", False),
        use_ltc=cfg.get("use_ltc", False)
    ).to(device)
    
    # Load weights
    if "rvq" in checkpoint and "talker" in checkpoint:
        rvq_state = strip_orig_mod(checkpoint["rvq"])
        talker_state = strip_orig_mod(checkpoint["talker"])
        rvq.load_state_dict(rvq_state, strict=False)
        talker.load_state_dict(talker_state, strict=False)
        print("✓ Loaded RVQ and Talker from checkpoint")
    elif "model" in checkpoint:
        # Try to split model state dict
        model_state = strip_orig_mod(checkpoint["model"])
        rvq_state = {k.replace("rvq.", ""): v for k, v in model_state.items() if k.startswith("rvq.")}
        talker_state = {k.replace("talker.", ""): v for k, v in model_state.items() if k.startswith("talker.")}
        if rvq_state:
            rvq.load_state_dict(rvq_state, strict=False)
        if talker_state:
            talker.load_state_dict(talker_state, strict=False)
        print("✓ Loaded from combined model state")
    else:
        # Try loading as-is
        state_dict = strip_orig_mod(checkpoint)
        try:
            rvq.load_state_dict({k: v for k, v in state_dict.items() if "rvq" in k or "codebook" in k}, strict=False)
            talker.load_state_dict({k: v for k, v in state_dict.items() if "talker" in k}, strict=False)
        except:
            print("⚠️  Warning: Could not load some weights")
    
    rvq.eval()
    talker.eval()
    
    print("✓ Models loaded successfully")
    print(f"  Codebooks: {codebooks}")
    print(f"  Codebook size: {codebook_size}")
    print(f"  Model dimension: {cfg.get('d_model', 256)}")
    
    return rvq, talker, cfg


def compute_codec_metrics(rvq, mel, device="cuda"):
    """
    Evaluate RVQ codec quality.
    
    Metrics:
    - Reconstruction error (MSE, MAE)
    - Codebook utilization
    - Compression ratio
    """
    rvq.eval()
    mel = mel.to(device)
    
    with torch.inference_mode():
        # Encode to discrete codes
        codes = rvq.encode(mel)  # (B, T, codebooks)
        
        # Decode back to mel
        mel_recon = rvq.decode(codes)  # (B, T, n_mels)
        
        # Reconstruction errors
        mse = torch.mean((mel - mel_recon) ** 2).item()
        mae = torch.mean(torch.abs(mel - mel_recon)).item()
        
        # Spectral convergence (normalized reconstruction error)
        numerator = torch.norm(mel - mel_recon, p='fro')
        denominator = torch.norm(mel, p='fro')
        spec_convergence = (numerator / denominator).item() if denominator > 0 else float('inf')
        
        # Codebook utilization (how many unique codes are used)
        unique_codes_per_book = []
        for book_idx in range(codes.shape[-1]):
            unique_codes = torch.unique(codes[:, :, book_idx]).numel()
            unique_codes_per_book.append(unique_codes)

        # Codebook perplexity (measures effective codebook usage uniformity)
        perplexity_per_book = []
        for book_idx in range(codes.shape[-1]):
            code_indices = codes[:, :, book_idx].reshape(-1).cpu().numpy()
            cb_size = rvq.codebooks[book_idx].num_embeddings if hasattr(rvq, 'codebooks') else int(code_indices.max()) + 1
            code_counts = np.bincount(code_indices, minlength=cb_size)
            probs = code_counts / code_counts.sum()
            probs = probs[probs > 0]  # remove zeros
            entropy = -np.sum(probs * np.log(probs))
            perplexity = np.exp(entropy)
            perplexity_per_book.append(float(perplexity))

        return {
            'mse': mse,
            'mae': mae,
            'spec_convergence': spec_convergence,
            'codes': codes,
            'mel_recon': mel_recon,
            'unique_codes_per_book': unique_codes_per_book,
            'perplexity_per_book': perplexity_per_book,
        }


def compute_ar_accuracy(talker, codes, device="cuda"):
    """
    Evaluate autoregressive prediction accuracy.
    
    Measures how well Talker predicts next codes given previous codes.
    """
    talker.eval()
    codes = codes.to(device)  # (B, T, codebooks)
    
    with torch.inference_mode():
        # Shift codes for teacher forcing
        # prev: codes shifted right by 1, with 0 at beginning
        prev = torch.roll(codes, 1, dims=1)
        prev[:, 0, :] = 0  # BOS
        
        # Forward pass
        base_logits, res_logits = talker(prev)  # Each: (B, T, codebook_size)
        
        # Compute accuracy for each codebook
        base_preds = base_logits.argmax(dim=-1)  # (B, T)
        res_preds = res_logits.argmax(dim=-1)    # (B, T)
        
        # Exclude first frame (BOS) from accuracy calculation
        base_acc = (base_preds[:, 1:] == codes[:, 1:, 0]).float().mean().item()
        res_acc = (res_preds[:, 1:] == codes[:, 1:, 1]).float().mean().item()
        
        # Top-5 accuracy
        base_top5_correct = 0
        res_top5_correct = 0
        total_frames = 0
        
        for b in range(codes.shape[0]):
            for t in range(1, codes.shape[1]):  # Skip BOS
                # Base codebook
                top5_base = base_logits[b, t].topk(5).indices
                if codes[b, t, 0] in top5_base:
                    base_top5_correct += 1
                
                # Residual codebook
                top5_res = res_logits[b, t].topk(5).indices
                if codes[b, t, 1] in top5_res:
                    res_top5_correct += 1
                
                total_frames += 1
        
        base_top5_acc = base_top5_correct / total_frames if total_frames > 0 else 0.0
        res_top5_acc = res_top5_correct / total_frames if total_frames > 0 else 0.0
        
        # Cross-entropy loss
        loss_fn = nn.CrossEntropyLoss(reduction='mean')
        base_loss = loss_fn(base_logits[:, :-1, :].reshape(-1, base_logits.size(-1)), 
                           codes[:, 1:, 0].reshape(-1)).item()
        res_loss = loss_fn(res_logits[:, :-1, :].reshape(-1, res_logits.size(-1)),
                          codes[:, 1:, 1].reshape(-1)).item()
        
        return {
            'base_accuracy': base_acc,
            'res_accuracy': res_acc,
            'base_top5_accuracy': base_top5_acc,
            'res_top5_accuracy': res_top5_acc,
            'base_loss': base_loss,
            'res_loss': res_loss,
            'total_loss': base_loss + res_loss,
        }


def evaluate_tts_quality(rvq, talker, cfg, device="cuda", num_samples=100, verbose=True):
    """
    Comprehensive TTS evaluation.
    
    Metrics:
    - Codec reconstruction quality
    - Autoregressive prediction accuracy
    - Codebook utilization
    """
    rvq.eval()
    talker.eval()
    
    tts_csv = cfg.get("tts_csv", "data/audio/production_tts.csv")
    if not os.path.exists(tts_csv):
        raise FileNotFoundError(f"TTS CSV not found: {tts_csv}")
    
    # Create dataset
    dataset = TTSDataset(
        csv_path=tts_csv,
        sr=cfg.get("sample_rate", 16000),
        n_mels=cfg.get("n_mels", 128),
        frame_ms=cfg.get("frame_ms", 80),
        cfg=cfg,
        shuffle_buffer_size=10000,
        seed=42,  # Fixed seed for reproducibility
        skip_samples=0
    )
    
    # Accumulators
    total_mse = 0.0
    total_mae = 0.0
    total_spec_conv = 0.0
    total_base_acc = 0.0
    total_res_acc = 0.0
    total_base_top5 = 0.0
    total_res_top5 = 0.0
    total_base_loss = 0.0
    total_res_loss = 0.0
    all_unique_codes = [[] for _ in range(cfg.get("codebooks", 2))]
    all_perplexities = [[] for _ in range(cfg.get("codebooks", 2))]
    num_valid = 0
    
    if verbose:
        print(f"\nEvaluating TTS quality on {num_samples} samples...")
        iterator = tqdm(dataset, total=num_samples, desc="Processing")
    else:
        iterator = iter(dataset)
    
    with torch.inference_mode():
        for i, mel in enumerate(iterator):
            if i >= num_samples:
                break
            
            try:
                mel = mel.unsqueeze(0).to(device)  # Add batch dimension
                
                # Evaluate codec
                codec_metrics = compute_codec_metrics(rvq, mel, device)
                
                # Evaluate AR model
                ar_metrics = compute_ar_accuracy(talker, codec_metrics['codes'], device)
                
                # Accumulate metrics
                total_mse += codec_metrics['mse']
                total_mae += codec_metrics['mae']
                total_spec_conv += codec_metrics['spec_convergence']
                total_base_acc += ar_metrics['base_accuracy']
                total_res_acc += ar_metrics['res_accuracy']
                total_base_top5 += ar_metrics['base_top5_accuracy']
                total_res_top5 += ar_metrics['res_top5_accuracy']
                total_base_loss += ar_metrics['base_loss']
                total_res_loss += ar_metrics['res_loss']
                
                # Track unique codes and perplexity
                for book_idx, unique_count in enumerate(codec_metrics['unique_codes_per_book']):
                    all_unique_codes[book_idx].append(unique_count)
                for book_idx, ppl in enumerate(codec_metrics['perplexity_per_book']):
                    all_perplexities[book_idx].append(ppl)
                
                num_valid += 1
                
            except Exception as e:
                if verbose:
                    print(f"\n⚠️  Error processing sample {i}: {e}")
                continue
    
    if num_valid == 0:
        raise ValueError("No valid samples processed!")
    
    # Calculate averages
    avg_mse = total_mse / num_valid
    avg_mae = total_mae / num_valid
    avg_spec_conv = total_spec_conv / num_valid
    avg_base_acc = total_base_acc / num_valid
    avg_res_acc = total_res_acc / num_valid
    avg_base_top5 = total_base_top5 / num_valid
    avg_res_top5 = total_res_top5 / num_valid
    avg_base_loss = total_base_loss / num_valid
    avg_res_loss = total_res_loss / num_valid
    
    # Codebook utilization statistics
    codebook_size = cfg.get("codebook_size", 128)
    avg_unique_per_book = [np.mean(codes) for codes in all_unique_codes]
    utilization_per_book = [avg / codebook_size for avg in avg_unique_per_book]
    avg_perplexity_per_book = [np.mean(ppls) for ppls in all_perplexities]

    return {
        'num_samples': num_valid,
        # Codec metrics
        'reconstruction_mse': avg_mse,
        'reconstruction_mae': avg_mae,
        'spectral_convergence': avg_spec_conv,
        # AR metrics
        'base_accuracy': avg_base_acc,
        'res_accuracy': avg_res_acc,
        'base_top5_accuracy': avg_base_top5,
        'res_top5_accuracy': avg_res_top5,
        'base_loss': avg_base_loss,
        'res_loss': avg_res_loss,
        'total_loss': avg_base_loss + avg_res_loss,
        # Codebook metrics
        'avg_unique_codes_per_book': avg_unique_per_book,
        'codebook_utilization': utilization_per_book,
        'codebook_perplexity': avg_perplexity_per_book,
    }


def print_results(metrics, cfg):
    """Pretty print evaluation results."""
    print(f"\n{'='*70}")
    print("TALKER (TTS) EVALUATION RESULTS")
    print(f"{'='*70}")
    
    print(f"\nSamples Evaluated: {metrics['num_samples']}")
    
    # Codec quality
    print(f"\nCODEC RECONSTRUCTION QUALITY:")
    print(f"  MSE: {metrics['reconstruction_mse']:.6f}")
    print(f"  MAE: {metrics['reconstruction_mae']:.6f}")
    print(f"  Spectral Convergence: {metrics['spectral_convergence']:.6f}")
    
    # AR prediction accuracy
    print(f"\nAUTOREGRESSIVE PREDICTION ACCURACY:")
    print(f"  Base Codebook:")
    print(f"    Top-1 Accuracy: {metrics['base_accuracy']*100:.2f}%")
    print(f"    Top-5 Accuracy: {metrics['base_top5_accuracy']*100:.2f}%")
    print(f"    Cross-Entropy Loss: {metrics['base_loss']:.4f}")
    print(f"  Residual Codebook:")
    print(f"    Top-1 Accuracy: {metrics['res_accuracy']*100:.2f}%")
    print(f"    Top-5 Accuracy: {metrics['res_top5_accuracy']*100:.2f}%")
    print(f"    Cross-Entropy Loss: {metrics['res_loss']:.4f}")
    print(f"  Total Loss: {metrics['total_loss']:.4f}")
    
    # Codebook utilization
    print(f"\nCODEBOOK UTILIZATION:")
    codebook_size = cfg.get("codebook_size", 128)
    for i, (unique, util) in enumerate(zip(metrics['avg_unique_codes_per_book'],
                                           metrics['codebook_utilization'])):
        print(f"  Codebook {i}: {unique:.1f}/{codebook_size} codes used ({util*100:.1f}%)")

    # Codebook perplexity
    print(f"\nCODEBOOK PERPLEXITY:")
    for i, ppl in enumerate(metrics['codebook_perplexity']):
        if ppl > 64:
            ppl_label = "EXCELLENT"
        elif ppl > 32:
            ppl_label = "GOOD"
        elif ppl > 16:
            ppl_label = "ACCEPTABLE"
        else:
            ppl_label = "POOR"
        print(f"  Codebook {i}: {ppl:.2f} / {codebook_size} ({ppl_label})")
    
    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print(f"{'='*70}")
    
    # Codec quality assessment
    mse = metrics['reconstruction_mse']
    if mse < 0.001:
        codec_status = "✓ EXCELLENT - Near-perfect reconstruction"
    elif mse < 0.01:
        codec_status = "✓ GOOD - High quality reconstruction"
    elif mse < 0.1:
        codec_status = "⚠ ACCEPTABLE - Moderate quality"
    else:
        codec_status = "✗ POOR - Needs more training"
    
    # AR accuracy assessment
    base_acc = metrics['base_accuracy']
    if base_acc > 0.70:
        ar_status = "✓ EXCELLENT - Strong autoregressive modeling"
    elif base_acc > 0.50:
        ar_status = "✓ GOOD - Reasonable prediction accuracy"
    elif base_acc > 0.30:
        ar_status = "⚠ ACCEPTABLE - Model is learning patterns"
    else:
        ar_status = "✗ POOR - Needs significant training"
    
    # Codebook utilization assessment
    avg_util = np.mean(metrics['codebook_utilization'])
    if avg_util > 0.7:
        util_status = "✓ EXCELLENT - Good codebook diversity"
    elif avg_util > 0.4:
        util_status = "✓ GOOD - Reasonable codebook usage"
    elif avg_util > 0.2:
        util_status = "⚠ ACCEPTABLE - Some codebook collapse"
    else:
        util_status = "✗ POOR - Severe codebook collapse"

    # Codebook perplexity assessment
    avg_ppl = np.mean(metrics['codebook_perplexity'])
    if avg_ppl > 64:
        ppl_status = "✓ EXCELLENT - Uniform codebook usage (perplexity {:.1f})".format(avg_ppl)
    elif avg_ppl > 32:
        ppl_status = "✓ GOOD - Reasonable codebook diversity (perplexity {:.1f})".format(avg_ppl)
    elif avg_ppl > 16:
        ppl_status = "⚠ ACCEPTABLE - Moderate codebook collapse (perplexity {:.1f})".format(avg_ppl)
    else:
        ppl_status = "✗ POOR - Severe codebook collapse (perplexity {:.1f})".format(avg_ppl)

    print(f"Codec Quality: {codec_status}")
    print(f"AR Prediction: {ar_status}")
    print(f"Codebook Usage: {util_status}")
    print(f"Codebook Perplexity: {ppl_status}")
    
    # Overall assessment
    print(f"\n{'='*70}")
    if mse < 0.01 and base_acc > 0.50 and avg_util > 0.4:
        print("✓ Talker is working well!")
        print("  Ready for TTS generation with good quality.")
    elif mse < 0.1 and base_acc > 0.30:
        print("⚠ Talker is learning but could improve.")
        print("  Consider training longer or adjusting hyperparameters.")
    else:
        print("✗ Talker needs more training.")
        print("  Current quality is not sufficient for production use.")
    
    print(f"{'='*70}")


def test_single_mel(rvq, talker, mel, cfg, device="cuda"):
    """Test on a single mel spectrogram."""
    print(f"\nTesting single mel spectrogram...")
    print(f"  Input shape: {mel.shape}")
    
    mel = mel.unsqueeze(0).to(device)
    
    # Evaluate codec
    codec_metrics = compute_codec_metrics(rvq, mel, device)
    
    # Evaluate AR model
    ar_metrics = compute_ar_accuracy(talker, codec_metrics['codes'], device)
    
    print(f"\nCodec Metrics:")
    print(f"  MSE: {codec_metrics['mse']:.6f}")
    print(f"  MAE: {codec_metrics['mae']:.6f}")
    print(f"  Codes shape: {codec_metrics['codes'].shape}")
    print(f"  Unique codes per book: {codec_metrics['unique_codes_per_book']}")
    
    print(f"\nAR Metrics:")
    print(f"  Base accuracy: {ar_metrics['base_accuracy']*100:.2f}%")
    print(f"  Res accuracy: {ar_metrics['res_accuracy']*100:.2f}%")
    print(f"  Total loss: {ar_metrics['total_loss']:.4f}")
    
    return codec_metrics, ar_metrics


def main():
    parser = argparse.ArgumentParser(description="Test Talker (TTS) with accuracy metrics")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/talker_tiny",
                       help="Path to Talker checkpoint directory")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate (default: 100)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with 10 samples")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config JSON (overrides checkpoint/auto-detected config)")
    parser.add_argument("--log_file", default=default_log_path(__file__), help="Write stdout/stderr to this file (UTF-8)")
    args = parser.parse_args()
    enable_log_file(args.log_file, header=f"test_talker.py start | checkpoint={args.checkpoint}")
    metrics_logger = MetricsLogger(
        script="test_talker.py",
        run_id=build_run_id("test_talker.py", args.config, args.checkpoint),
        metrics_path=os.path.join("logs", "metrics", "test_talker.jsonl"),
        device=args.device,
    )
    metrics_logger.event(epoch=0, batch=0, step=0, name="run_start", value=1.0, extra={"is_resume": False, "resume_from_step": 0})
    
    if args.quick:
        args.num_samples = 10
    
    print("=" * 70)
    print("TALKER (TTS) ACCURACY TEST")
    print("=" * 70)

    if args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Load model
    try:
        rvq, talker, cfg = load_model_and_codec(args.checkpoint, args.device, config_path=args.config)
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate on dataset
    try:
        metrics = evaluate_tts_quality(
            rvq, talker, cfg,
            device=args.device,
            num_samples=args.num_samples,
            verbose=True
        )
        metrics_logger.test_metrics(metrics)
        print_results(metrics, cfg)
        
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()