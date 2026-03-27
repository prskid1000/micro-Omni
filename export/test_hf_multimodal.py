"""
Test HuggingFace multimodal model (MuOmniMultimodalModel) with scored analytics.
Tests: text, image encoding, audio encoding, image+text, audio+text, all combined.
Uses random samples from training data.

Usage:
    python export/test_hf_multimodal.py
    python export/test_hf_multimodal.py --num_samples 20 --device cpu
"""

import sys
import os
import argparse
import random
import json
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
torch.set_float32_matmul_precision('high')

from modeling_muomni import MuOmniMultimodalModel
from PIL import Image
from torchvision import transforms
import torchaudio

# Add parent dir for omni imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omni.tokenizer import BPETokenizer
from omni.io_utils import load_audio, enable_log_file, default_log_path
from omni.eval_utils import sample_lines, count_topk_from_logits, safe_perplexity


def main():
    parser = argparse.ArgumentParser(description="Test HF multimodal model")
    parser.add_argument("--model_dir", default="export", help="Export directory")
    parser.add_argument("--corpus", default="data/text/production_corpus.txt")
    parser.add_argument("--image_manifest", default="data/images/production_annotations.json")
    parser.add_argument("--image_root", default="data/images")
    parser.add_argument("--audio_csv", default="data/audio/production_asr.csv")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_file", default=default_log_path(__file__), help="Write stdout/stderr to this file (UTF-8)")
    args = parser.parse_args()
    enable_log_file(args.log_file, header=f"test_hf_multimodal.py start | model_dir={args.model_dir}")

    print("=" * 60)
    print("HF MULTIMODAL MODEL — SCORED TEST")
    print("=" * 60)

    model = MuOmniMultimodalModel.from_pretrained_safetensors(args.model_dir)
    model = model.to(args.device).eval()
    tok = BPETokenizer(os.path.join(args.model_dir, "tokenizer.model"))

    # Component params
    total_params = 0
    for name in ["thinker", "audio_encoder", "vision_encoder", "talker", "rvq", "proj_a", "proj_v"]:
        comp = getattr(model, name, None)
        if comp:
            p = sum(p.numel() for p in comp.parameters())
            print(f"  {name}: {p:,}")
            total_params += p
    print(f"  TOTAL: {total_params:,}")

    random.seed(args.seed)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    img_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    # ---- 1. TEXT-ONLY ----
    print("\n" + "-" * 40)
    print("1. TEXT-ONLY FORWARD")
    print("-" * 40)
    text_samples = sample_lines(args.corpus, args.num_samples, min_len=11, seed=args.seed)
    c1, c5, total, total_loss = 0, 0, 0, 0.0
    for line in text_samples:
        ids = [1] + tok.encode(line)
        if len(ids) < 4:
            continue
        x = torch.tensor([ids[:-1]], device=args.device)
        tgt = torch.tensor([ids[1:]], device=args.device)
        with torch.inference_mode():
            logits = model(input_ids=x).logits
        total_loss += loss_fn(logits[0], tgt[0]).item()
        topk = count_topk_from_logits(logits[0], tgt[0], ks=(1, 5))
        c1 += topk[1]
        c5 += topk[5]
        total += tgt.numel()
    avg_loss = total_loss / max(total, 1)
    ppl = safe_perplexity(avg_loss, max_exp_input=20.0)
    print(f"  Tokens: {total}")
    print(f"  Loss: {avg_loss:.4f}, Perplexity: {ppl:.2f}")
    print(f"  Top-1: {c1/total*100:.2f}%, Top-5: {c5/total*100:.2f}%")

    # ---- 2. IMAGE ENCODING ----
    print("\n" + "-" * 40)
    print("2. IMAGE ENCODING")
    print("-" * 40)
    manifest = json.load(open(args.image_manifest))
    img_items = random.sample(manifest, min(args.num_samples, len(manifest)))
    embeddings = []
    for item in img_items:
        img_path = os.path.join(args.image_root, item["image"])
        if not os.path.exists(img_path):
            continue
        img = img_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(args.device)
        with torch.inference_mode():
            emb = model.encode_image(img)
        embeddings.append(emb.squeeze())
    if embeddings:
        emb_stack = torch.stack(embeddings)
        emb_norm = emb_stack / emb_stack.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sim = emb_norm @ emb_norm.T
        diversity = 1.0 - sim.fill_diagonal_(0).mean().item()
        print(f"  Samples: {len(embeddings)}, Dim: {embeddings[0].shape}")
        print(f"  Diversity: {diversity:.4f} ({'EXCELLENT' if diversity > 0.7 else 'GOOD' if diversity > 0.4 else 'POOR'})")

    # ---- 3. AUDIO ENCODING ----
    print("\n" + "-" * 40)
    print("3. AUDIO ENCODING")
    print("-" * 40)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=160, win_length=400, n_mels=128
    )
    asr_rows = list(csv.reader(open(args.audio_csv)))[1:]
    aud_items = random.sample(asr_rows, min(args.num_samples, len(asr_rows)))
    aud_embeddings = []
    for row in aud_items:
        wav_path = row[0]
        if not os.path.exists(wav_path):
            continue
        try:
            wav, sr = load_audio(wav_path)
            wav = wav.to(args.device)
            mel = mel_transform.to(args.device)(wav).squeeze(0).T.unsqueeze(0)  # (1, T, 128)
            with torch.inference_mode():
                emb = model.encode_audio(mel)
            aud_embeddings.append(emb.squeeze(0).mean(0))
        except Exception as e:
            continue
    if aud_embeddings:
        aud_stack = torch.stack(aud_embeddings)
        aud_norm = aud_stack / aud_stack.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        aud_sim = aud_norm @ aud_norm.T
        aud_diversity = 1.0 - aud_sim.fill_diagonal_(0).mean().item()
        print(f"  Samples: {len(aud_embeddings)}")
        print(f"  Diversity: {aud_diversity:.4f} ({'EXCELLENT' if aud_diversity > 0.7 else 'GOOD' if aud_diversity > 0.4 else 'POOR'})")

    # ---- 4. IMAGE + TEXT ----
    print("\n" + "-" * 40)
    print("4. IMAGE + TEXT (multimodal)")
    print("-" * 40)
    mm_loss, mm_count = 0.0, 0
    for item in img_items[:10]:
        img_path = os.path.join(args.image_root, item["image"])
        if not os.path.exists(img_path):
            continue
        caption = item.get("caption", "")
        if not caption:
            continue
        img = img_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(args.device)
        ids = [1] + tok.encode(caption)
        x = torch.tensor([ids[:-1]], device=args.device)
        tgt = torch.tensor([ids[1:]], device=args.device)
        with torch.inference_mode():
            out = model(input_ids=x, pixel_values=img)
        logits = out.logits[:, -x.shape[1]:, :]
        if logits.shape[1] == tgt.shape[1]:
            loss = loss_fn(logits[0], tgt[0]).item() / tgt.shape[1]
            mm_loss += loss
            mm_count += 1
    if mm_count > 0:
        avg_mm = mm_loss / mm_count
        mm_ppl = safe_perplexity(avg_mm, max_exp_input=20.0)
        print(f"  Samples: {mm_count}")
        print(f"  Loss: {avg_mm:.4f}, Perplexity: {mm_ppl:.2f}")

    # ---- 5. AUDIO + TEXT ----
    print("\n" + "-" * 40)
    print("5. AUDIO + TEXT (multimodal)")
    print("-" * 40)
    at_loss, at_count = 0.0, 0
    for row in aud_items[:10]:
        wav_path, text = row[0], row[1] if len(row) > 1 else ""
        if not os.path.exists(wav_path) or not text:
            continue
        try:
            wav, sr = load_audio(wav_path)
            wav = wav.to(args.device)
            mel = mel_transform.to(args.device)(wav).squeeze(0).T.unsqueeze(0)
            ids = [1] + tok.encode(text)
            x = torch.tensor([ids[:-1]], device=args.device)
            tgt = torch.tensor([ids[1:]], device=args.device)
            with torch.inference_mode():
                out = model(input_ids=x, mel_spectrogram=mel)
            logits = out.logits[:, -x.shape[1]:, :]
            if logits.shape[1] == tgt.shape[1]:
                loss = loss_fn(logits[0], tgt[0]).item() / tgt.shape[1]
                at_loss += loss
                at_count += 1
        except Exception:
            continue
    if at_count > 0:
        avg_at = at_loss / at_count
        at_ppl = safe_perplexity(avg_at, max_exp_input=20.0)
        print(f"  Samples: {at_count}")
        print(f"  Loss: {avg_at:.4f}, Perplexity: {at_ppl:.2f}")

    # ---- SUMMARY ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Text Top-1:          {c1/total*100:.2f}%")
    print(f"  Text Top-5:          {c5/total*100:.2f}%")
    print(f"  Text Perplexity:     {ppl:.2f}")
    if embeddings:
        print(f"  Image Diversity:     {diversity:.4f}")
    if aud_embeddings:
        print(f"  Audio Diversity:     {aud_diversity:.4f}")
    if mm_count > 0:
        print(f"  Image+Text PPL:      {mm_ppl:.2f}")
    if at_count > 0:
        print(f"  Audio+Text PPL:      {at_ppl:.2f}")
    print(f"  Total Params:        {total_params:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
