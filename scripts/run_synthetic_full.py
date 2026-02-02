"""
Full workflow: create synthetic datasets → train all models → run tests.
Use for quick validation on synthetic data (fits 12GB VRAM).
"""

import os
import sys
import subprocess
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, desc=""):
    print(f"\n{'='*60}")
    if desc:
        print(f"  {desc}")
    print(f"  $ {cmd}")
    print("="*60)
    result = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\n[FAIL] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


def main():
    parser = argparse.ArgumentParser(description="Create synthetic data, train, and test μOmni")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Samples per modality for quick run (default: 1000)")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip dataset creation (use existing data)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (only create data and test)")
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip tests (only create data and train)")
    parser.add_argument("--stages", type=str, default="A,B,C,D,E,G",
                        help="Training stages to run: A=Thinker, B=AudioEnc, C=Vision, D=Talker, E=SFT, F=Vocoder, G=OCR (default: A,B,C,D,E,G)")
    args = parser.parse_args()

    stages = set(s.strip().upper() for s in args.stages.split(",") if s.strip())

    # 1. Create synthetic datasets
    if not args.skip_data:
        run(
            f"python scripts/make_synthetic_datasets.py --num-samples {args.num_samples}",
            "Creating synthetic datasets"
        )
    else:
        print("\nSkipping dataset creation (--skip-data)")

    # 2. Train
    if not args.skip_train:
        if "A" in stages:
            run(
                "python train_text.py --config configs/synthetic_thinker.json",
                "Stage A: Thinker (text LLM)"
            )
        if "B" in stages:
            run(
                "python train_audio_enc.py --config configs/synthetic_audio_enc.json",
                "Stage B: Audio encoder (ASR)"
            )
        if "C" in stages:
            run(
                "python train_vision.py --config configs/synthetic_vision.json",
                "Stage C: Vision encoder"
            )
        if "D" in stages:
            run(
                "python train_talker.py --config configs/synthetic_talker.json",
                "Stage D: Talker + RVQ codec"
            )
        if "E" in stages:
            run(
                "python sft_omni.py --config configs/synthetic_omni_sft.json",
                "Stage E: Omni SFT (multimodal)"
            )
        if "F" in stages:
            run(
                "python train_vocoder.py --config configs/synthetic_vocoder.json",
                "Stage F: HiFi-GAN vocoder (optional)"
            )
        if "G" in stages:
            run(
                "python train_ocr.py --config configs/synthetic_ocr.json",
                "Stage G: OCR model"
            )
    else:
        print("\nSkipping training (--skip-train)")

    # 3. Run tests
    if not args.skip_test:
        tests = [
            ("test_thinker.py", "Thinker", "checkpoints/thinker_tiny"),
            ("test_audio_enc.py", "Audio encoder", "checkpoints/audio_enc_tiny"),
            ("test_vision.py", "Vision encoder", "checkpoints/vision_tiny"),
            ("test_talker.py", "Talker", "checkpoints/talker_tiny"),
            ("test_ocr.py", "OCR", "checkpoints/ocr_tiny"),
            ("test_vocoder.py", "Vocoder", "checkpoints/vocoder_tiny"),  # Optional - skip if not trained
        ]
        for script, name, ckpt in tests:
            # Skip vocoder test if checkpoint missing (Stage F is optional)
            if "vocoder" in script.lower() and not os.path.exists(os.path.join(PROJECT_ROOT, ckpt, "vocoder.pt")):
                print(f"\n[SKIP] {name}: checkpoint not found (Stage F optional, run with --stages A,B,C,D,E,F,G to train)")
                continue
            run(
                f"python {script} --checkpoint {ckpt}",
                f"Test: {name}"
            )
    else:
        print("\nSkipping tests (--skip-test)")

    print("\n" + "="*60)
    print("[OK] Full workflow completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
