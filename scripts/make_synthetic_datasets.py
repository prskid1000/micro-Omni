"""
Create synthetic datasets for all μOmni models.
Outputs to production paths so configs work without changes.
"""

import os, csv, random, json, wave, struct, math, argparse
from PIL import Image, ImageDraw, ImageFont


def main(num_samples=10000, base='data'):
    os.makedirs(base, exist_ok=True)
    random.seed(42)

    subjects = ["cat", "dog", "bird", "car", "tree", "house", "computer", "book", "phone", "person"]
    verbs = ["runs", "jumps", "flies", "drives", "grows", "stands", "processes", "reads", "rings", "walks"]
    objects = ["quickly", "high", "fast", "tall", "firmly", "efficiently", "carefully", "loudly", "slowly", "gracefully"]
    topics = ["science", "technology", "nature", "art", "music", "sports", "food", "travel", "education", "health"]

    # Text: diverse sentences
    print(f"Generating {num_samples} text samples...")
    os.makedirs(f'{base}/text', exist_ok=True)
    with open(f'{base}/text/production_corpus.txt', 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            if i % 4 == 0:
                s, v, o = random.choice(subjects), random.choice(verbs), random.choice(objects)
                f.write(f"The {s} {v} {o}.\n")
            elif i % 4 == 1:
                topic = random.choice(topics)
                f.write(f"Learning about {topic} is fascinating and important for understanding the world.\n")
            elif i % 4 == 2:
                color = random.choice(["red", "blue", "green", "yellow", "purple", "orange"])
                shape = random.choice(["circle", "square", "triangle", "rectangle"])
                f.write(f"A {color} {shape} appears in the image with interesting details.\n")
            else:
                num = random.randint(1, 1000)
                f.write(f"Sample number {num} demonstrates various linguistic patterns and structures.\n")

    # Images: diverse geometric compositions (image_root=data/images, images in images/ subdir)
    print(f"Generating {num_samples} image samples...")
    img_root = f'{base}/images/images'
    os.makedirs(img_root, exist_ok=True)
    ann = []
    shapes = ["square", "circle", "rectangle", "triangle", "ellipse"]
    colors_bg = [(0,0,255), (255,0,0), (0,255,0), (255,255,0), (255,0,255), (0,255,255), (128,128,128), (255,165,0)]
    colors_fg = [(255,0,0), (0,255,0), (255,255,0), (0,0,255), (255,0,255), (0,255,255), (255,255,255), (0,0,0)]

    for i in range(num_samples):
        W = H = 224
        bg_color = random.choice(colors_bg)
        img = Image.new('RGB', (W, H), bg_color)
        draw = ImageDraw.Draw(img)
        shape_type = random.choice(shapes)
        size = random.randint(30, 150)
        x = random.randint(10, W - size - 10)
        y = random.randint(10, H - size - 10)
        fg_color = random.choice(colors_fg)
        if shape_type == "square":
            draw.rectangle([x, y, x+size, y+size], fill=fg_color)
            caption = f"A {fg_color} square of size {size}px on a {bg_color} background."
        elif shape_type == "circle":
            draw.ellipse([x, y, x+size, y+size], fill=fg_color)
            caption = f"A {fg_color} circle with diameter {size}px on a {bg_color} background."
        elif shape_type == "rectangle":
            w, h = size, size // 2
            draw.rectangle([x, y, x+w, y+h], fill=fg_color)
            caption = f"A {fg_color} rectangle {w}x{h}px on a {bg_color} background."
        elif shape_type == "triangle":
            points = [(x+size//2, y), (x, y+size), (x+size, y+size)]
            draw.polygon(points, fill=fg_color)
            caption = f"A {fg_color} triangle with base {size}px on a {bg_color} background."
        else:
            w, h = size, size // 2
            draw.ellipse([x, y, x+w, y+h], fill=fg_color)
            caption = f"An {fg_color} ellipse {w}x{h}px on a {bg_color} background."
        img.save(f'{img_root}/{i:06d}.png')
        ann.append({"image": f"images/{i:06d}.png", "caption": caption})

    with open(f'{base}/images/production_annotations.json', 'w', encoding='utf-8') as f:
        json.dump(ann, f)

    # Audio: synthetic audio with varied patterns
    print(f"Generating {num_samples} audio samples...")
    aud_dir = f'{base}/audio/wav'
    os.makedirs(aud_dir, exist_ok=True)
    asr_rows = []
    tts_rows = []
    sr = 16000

    for i in range(num_samples):
        dur = 0.5 + random.random() * 2.5
        n = int(sr * dur)
        freq = 200 + random.randint(0, 400)
        wav_path = f"data/audio/wav/{i:06d}.wav"
        with wave.open(os.path.join(base, "audio", "wav", f"{i:06d}.wav"), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            pattern = i % 5
            for t in range(n):
                if pattern == 0:
                    val = int(32767 * 0.2 * math.sin(2 * math.pi * freq * t / sr))
                elif pattern == 1:
                    val = int(32767 * 0.15 * (math.sin(2 * math.pi * freq * t / sr) +
                                              0.5 * math.sin(4 * math.pi * freq * t / sr)))
                elif pattern == 2:
                    val = int(32767 * 0.2 * (1 if math.sin(2 * math.pi * freq * t / sr) > 0 else -1))
                elif pattern == 3:
                    f = freq + (t / n) * 200
                    val = int(32767 * 0.2 * math.sin(2 * math.pi * f * t / sr))
                else:
                    mod = 1 + 0.3 * math.sin(2 * math.pi * 5 * t / sr)
                    val = int(32767 * 0.2 * mod * math.sin(2 * math.pi * freq * t / sr))
                wf.writeframes(struct.pack('<h', val))
        # ASR: short transcriptions for CTC (need output_frames >= len(text); with 4x downsample,
        # 0.5 sec = ~12 output frames, so keep transcriptions <= 10 chars for shortest clips)
        asr_text = f"sample {i}"
        # TTS: can use longer text for variety
        tts_text = f"synthetic audio sample {i} with frequency {freq} hertz and duration {dur:.2f} seconds"
        asr_rows.append([wav_path, asr_text])
        tts_rows.append([tts_text, wav_path])

    with open(f'{base}/audio/production_asr.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["wav", "text"])
        w.writerows(asr_rows)
    with open(f'{base}/audio/production_tts.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["text", "wav"])
        w.writerows(tts_rows)

    # OCR: images with diverse text (image_root=data/ocr, images in images/ subdir)
    print(f"Generating {num_samples} OCR samples...")
    ocr_dir = f'{base}/ocr'
    ocr_img_dir = f'{ocr_dir}/images'
    os.makedirs(ocr_img_dir, exist_ok=True)
    ocr_rows = []
    words = ["HELLO", "WORLD", "TEST", "OCR", "TEXT", "IMAGE", "DATA", "AI", "ML", "NLP",
             "PYTHON", "CODE", "TRAIN", "MODEL", "NEURAL", "NET", "DEEP", "LEARN", "SMART", "FAST"]
    numbers = ["123", "456", "789", "2024", "2025", "100", "999", "42", "007", "8080"]
    phrases = ["HELLO WORLD", "TEST OCR", "AI ML", "PYTHON CODE", "NEURAL NET",
                "DEEP LEARNING", "SMART AI", "FAST ML", "TRAIN MODEL", "DATA TEXT"]
    mixed = ["ABC123", "XYZ789", "AI2024", "ML42", "NET100", "CODE007", "TEXT999", "DATA8080"]
    all_texts = words + numbers + phrases + mixed

    for i in range(num_samples):
        W = H = 224
        bg_colors = [(255,255,255), (240,240,240), (200,200,200), (255,250,240), (240,255,240)]
        bg = random.choice(bg_colors)
        img = Image.new('RGB', (W, H), bg)
        draw = ImageDraw.Draw(img)
        text = random.choice(all_texts)
        font_size = random.choice([32, 40, 48, 56, 64])
        try:
            if os.path.exists("C:/Windows/Fonts/arial.ttf"):
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
            elif os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = random.randint(10, max(10, W - text_width - 10))
        y = random.randint(10, max(10, H - text_height - 10))
        text_colors = [(0,0,0), (50,50,50), (100,0,0), (0,100,0), (0,0,100), (100,50,0), (0,50,100)]
        color = random.choice(text_colors)
        draw.text((x, y), text, fill=color, font=font)
        img.save(f'{ocr_img_dir}/{i:06d}.png')
        ocr_rows.append([f"images/{i:06d}.png", text])

    with open(f'{ocr_dir}/production_ocr.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["image", "text"])
        w.writerows(ocr_rows)

    print("\n[OK] Synthetic datasets created under data/ (production paths).")
    print(f"  - Text: data/text/production_corpus.txt ({num_samples} samples)")
    print(f"  - Images: data/images/production_annotations.json ({num_samples} samples)")
    print(f"  - Audio: data/audio/production_asr.csv, production_tts.csv ({num_samples} samples each)")
    print(f"  - OCR: data/ocr/production_ocr.csv ({num_samples} samples)")
    print("\nUse synthetic configs (vocab_size=1024, lower max_steps for small data):")
    print("  python train_text.py --config configs/synthetic_thinker.json")
    print("  python train_audio_enc.py --config configs/synthetic_audio_enc.json")
    print("  python train_vision.py --config configs/synthetic_vision.json")
    print("  python train_talker.py --config configs/synthetic_talker.json")
    print("  python train_ocr.py --config configs/synthetic_ocr.json")
    print("  python sft_omni.py --config configs/synthetic_omni_sft.json  # after A-D trained")
    print("  python train_vocoder.py --config configs/synthetic_vocoder.json  # optional")
    print("\nOr run full workflow: python scripts/run_synthetic_full.py --num-samples 1000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create synthetic datasets for μOmni")
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="Number of samples per modality (default: 10000)")
    parser.add_argument("--base", type=str, default="data",
                        help="Base data directory (default: data)")
    args = parser.parse_args()
    main(num_samples=args.num_samples, base=args.base)
