"""
Generate high-quality synthetic training data for all modalities.
Produces 5K samples per modality with maximum diversity.

Usage:
    python generate_synthetic_data.py --all              # Generate everything
    python generate_synthetic_data.py --text             # Text corpus only
    python generate_synthetic_data.py --audio            # ASR + TTS audio
    python generate_synthetic_data.py --images           # Image + captions
    python generate_synthetic_data.py --ocr              # OCR images + text
    python generate_synthetic_data.py --count 10000      # Override sample count
"""

import argparse
import csv
import json
import math
import os
import random
import struct
import wave
from pathlib import Path

# ============================================================
# 1. TEXT CORPUS GENERATOR
# ============================================================

def generate_text_corpus(output_path: str, count: int = 5000):
    """Generate diverse text corpus using CFG-like rules + Faker-style entities."""
    print(f"Generating {count} text samples...")

    # Rich vocabulary pools
    subjects = [
        "the cat", "a dog", "the bird", "my friend", "the teacher", "a student",
        "the doctor", "a child", "the farmer", "an artist", "the scientist",
        "a musician", "the chef", "a pilot", "the dancer", "a writer",
        "the old man", "a young woman", "the little boy", "a tall girl",
        "the clever fox", "a brave knight", "the wise owl", "a tiny mouse",
        "the happy family", "a busy worker", "the kind nurse", "a strong lion",
        "the quiet librarian", "a fast runner", "the gentle giant", "a curious baby"
    ]

    verbs_intransitive = [
        "runs", "walks", "sleeps", "sings", "dances", "jumps", "swims",
        "flies", "laughs", "cries", "smiles", "thinks", "waits", "reads",
        "writes", "plays", "works", "rests", "dreams", "travels"
    ]

    verbs_transitive = [
        "sees", "likes", "loves", "finds", "takes", "gives", "makes",
        "builds", "reads", "writes", "draws", "paints", "cooks", "eats",
        "drinks", "carries", "holds", "opens", "closes", "breaks"
    ]

    objects = [
        "a book", "the ball", "some food", "a letter", "the door", "a picture",
        "the cake", "a song", "the box", "a flower", "the key", "a map",
        "the bridge", "a boat", "the house", "a garden", "the window",
        "a present", "the clock", "a puzzle", "the hat", "a coin", "the rope"
    ]

    locations = [
        "in the park", "at home", "near the river", "on the hill", "by the lake",
        "in the garden", "at school", "in the city", "on the farm", "near the forest",
        "at the market", "in the kitchen", "on the roof", "by the sea", "in the cave",
        "at the library", "in the hospital", "on the road", "near the castle"
    ]

    times = [
        "today", "yesterday", "every morning", "at night", "in the summer",
        "on Monday", "last week", "this afternoon", "before sunrise", "after lunch",
        "during winter", "at noon", "in the evening", "early morning", "late at night"
    ]

    adjectives = [
        "big", "small", "red", "blue", "green", "old", "new", "fast", "slow",
        "bright", "dark", "warm", "cold", "soft", "hard", "sweet", "loud",
        "quiet", "tall", "short", "heavy", "light", "clean", "dirty", "sharp"
    ]

    conjunctions = ["and", "but", "so", "because", "although", "when", "while", "if"]

    adverbs = [
        "quickly", "slowly", "carefully", "happily", "sadly", "loudly", "quietly",
        "always", "never", "often", "sometimes", "suddenly", "gently", "bravely"
    ]

    # Sentence templates (30+ patterns for diversity)
    def gen_simple_svo():
        return f"{random.choice(subjects)} {random.choice(verbs_transitive)} {random.choice(objects)}."

    def gen_simple_sv_loc():
        return f"{random.choice(subjects)} {random.choice(verbs_intransitive)} {random.choice(locations)}."

    def gen_adverb():
        return f"{random.choice(subjects)} {random.choice(adverbs)} {random.choice(verbs_intransitive)} {random.choice(locations)}."

    def gen_time():
        return f"{random.choice(times)}, {random.choice(subjects)} {random.choice(verbs_intransitive)}."

    def gen_compound():
        s1 = f"{random.choice(subjects)} {random.choice(verbs_intransitive)}"
        s2 = f"{random.choice(subjects)} {random.choice(verbs_intransitive)}"
        return f"{s1} {random.choice(conjunctions)} {s2}."

    def gen_complex():
        s1 = f"{random.choice(subjects)} {random.choice(verbs_transitive)} {random.choice(objects)}"
        s2 = f"{random.choice(subjects)} {random.choice(verbs_intransitive)} {random.choice(locations)}"
        return f"{random.choice(conjunctions).capitalize()} {s2}, {s1}."

    def gen_question():
        templates = [
            f"Does {random.choice(subjects)} {random.choice(verbs_intransitive)}?",
            f"Where does {random.choice(subjects)} {random.choice(verbs_intransitive)}?",
            f"What does {random.choice(subjects)} {random.choice(verbs_transitive)}?",
            f"Can {random.choice(subjects)} {random.choice(verbs_intransitive)} {random.choice(locations)}?",
            f"Why does {random.choice(subjects)} {random.choice(verbs_intransitive)} {random.choice(adverbs)}?",
            f"How many {random.choice(['cats', 'dogs', 'birds', 'fish', 'trees'])} are {random.choice(locations)}?",
        ]
        return random.choice(templates)

    def gen_imperative():
        templates = [
            f"Please {random.choice(verbs_intransitive)} {random.choice(locations)}.",
            f"Do not {random.choice(verbs_intransitive)} {random.choice(locations)}.",
            f"Try to {random.choice(verbs_transitive)} {random.choice(objects)} {random.choice(adverbs)}.",
            f"Always {random.choice(verbs_intransitive)} {random.choice(adverbs)}.",
        ]
        return random.choice(templates)

    def gen_comparison():
        a, b = random.sample(subjects, 2)
        adj = random.choice(adjectives)
        return f"{a} is more {adj} than {b}."

    def gen_counting():
        n = random.randint(1, 20)
        nums = " ".join(str(i) for i in range(1, n + 1))
        return f"Count: {nums}."

    def gen_math():
        a, b = random.randint(1, 50), random.randint(1, 50)
        ops = [("+", a + b), ("-", abs(a - b)), ("times", a * b)]
        op, result = random.choice(ops)
        return f"{a} {op} {b} is {result}."

    def gen_list():
        items = random.sample(["apples", "bread", "milk", "eggs", "rice", "fish",
                               "cheese", "water", "salt", "sugar", "butter", "flour",
                               "meat", "tea", "coffee", "juice", "soup", "cake"], random.randint(3, 6))
        return f"I need {', '.join(items[:-1])} and {items[-1]}."

    def gen_description():
        subj = random.choice(subjects)
        adjs = random.sample(adjectives, random.randint(1, 3))
        return f"{subj} is {' and '.join(adjs)}."

    def gen_possessive():
        owner = random.choice(["my", "your", "his", "her", "their", "our"])
        item = random.choice(["house", "car", "book", "garden", "friend", "dog", "idea", "plan"])
        adj = random.choice(adjectives)
        return f"{owner.capitalize()} {item} is very {adj}."

    def gen_there_is():
        n = random.randint(1, 10)
        thing = random.choice(["cats", "birds", "trees", "houses", "books", "flowers", "stars", "clouds"])
        loc = random.choice(locations)
        return f"There are {n} {thing} {loc}."

    def gen_dialogue():
        names = ["Alice", "Bob", "Sam", "Emma", "John", "Mary", "Tom", "Lisa"]
        n1, n2 = random.sample(names, 2)
        phrases = [
            f'{n1} said "hello" to {n2}.',
            f'{n1} asked {n2} to {random.choice(verbs_intransitive)}.',
            f'{n1} and {n2} {random.choice(verbs_intransitive)} together.',
            f'{n1} told {n2} about {random.choice(objects)}.',
        ]
        return random.choice(phrases)

    def gen_conditional():
        return f"If {random.choice(subjects)} {random.choice(verbs_intransitive)}, then {random.choice(subjects)} will {random.choice(verbs_intransitive)} too."

    def gen_sequence():
        steps = random.randint(2, 4)
        actions = random.sample(verbs_intransitive, steps)
        subj = random.choice(subjects)
        seq = ", then ".join(f"{a}s" for a in actions)
        return f"First {subj} {seq}."

    generators = [
        gen_simple_svo, gen_simple_sv_loc, gen_adverb, gen_time,
        gen_compound, gen_complex, gen_question, gen_imperative,
        gen_comparison, gen_counting, gen_math, gen_list,
        gen_description, gen_possessive, gen_there_is, gen_dialogue,
        gen_conditional, gen_sequence,
    ]

    sentences = set()
    while len(sentences) < count:
        gen = random.choice(generators)
        sent = gen()
        # Capitalize first letter, basic cleanup
        sent = sent[0].upper() + sent[1:]
        sentences.add(sent)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in sorted(sentences):
            f.write(s + "\n")

    print(f"  Written {len(sentences)} unique sentences to {output_path}")
    # Show sample
    samples = list(sentences)[:5]
    for s in samples:
        print(f"    {s}")


# ============================================================
# 2. IMAGE + CAPTION GENERATOR
# ============================================================

def generate_images(output_dir: str, manifest_path: str, count: int = 5000):
    """Generate diverse synthetic images with descriptive captions."""
    from PIL import Image, ImageDraw, ImageFont
    print(f"Generating {count} image+caption pairs...")

    os.makedirs(output_dir, exist_ok=True)
    IMG_SIZE = 224

    color_names = {
        (255, 0, 0): "red", (0, 255, 0): "green", (0, 0, 255): "blue",
        (255, 255, 0): "yellow", (255, 128, 0): "orange", (128, 0, 255): "purple",
        (255, 192, 203): "pink", (0, 255, 255): "cyan", (255, 255, 255): "white",
        (128, 128, 128): "gray", (0, 128, 0): "dark green", (0, 0, 128): "navy",
        (139, 69, 19): "brown", (255, 215, 0): "gold", (0, 128, 128): "teal",
    }
    colors = list(color_names.keys())

    bg_types = ["solid", "gradient_h", "gradient_v", "checkerboard", "stripes"]
    shape_types = ["circle", "rectangle", "triangle", "ellipse", "diamond", "star", "line"]
    size_names = {0: "tiny", 1: "small", 2: "medium", 3: "large"}
    pos_names = {
        (0, 0): "top-left", (1, 0): "top-center", (2, 0): "top-right",
        (0, 1): "middle-left", (1, 1): "center", (2, 1): "middle-right",
        (0, 2): "bottom-left", (1, 2): "bottom-center", (2, 2): "bottom-right",
    }

    def make_background(draw, img, bg_type, c1, c2):
        if bg_type == "solid":
            draw.rectangle([0, 0, IMG_SIZE, IMG_SIZE], fill=c1)
        elif bg_type == "gradient_h":
            for x in range(IMG_SIZE):
                r = int(c1[0] + (c2[0] - c1[0]) * x / IMG_SIZE)
                g = int(c1[1] + (c2[1] - c1[1]) * x / IMG_SIZE)
                b = int(c1[2] + (c2[2] - c1[2]) * x / IMG_SIZE)
                draw.line([(x, 0), (x, IMG_SIZE)], fill=(r, g, b))
        elif bg_type == "gradient_v":
            for y in range(IMG_SIZE):
                r = int(c1[0] + (c2[0] - c1[0]) * y / IMG_SIZE)
                g = int(c1[1] + (c2[1] - c1[1]) * y / IMG_SIZE)
                b = int(c1[2] + (c2[2] - c1[2]) * y / IMG_SIZE)
                draw.line([(0, y), (IMG_SIZE, y)], fill=(r, g, b))
        elif bg_type == "checkerboard":
            sq = random.choice([16, 28, 32, 56])
            for y in range(0, IMG_SIZE, sq):
                for x in range(0, IMG_SIZE, sq):
                    fill = c1 if (x // sq + y // sq) % 2 == 0 else c2
                    draw.rectangle([x, y, x + sq, y + sq], fill=fill)
        elif bg_type == "stripes":
            w = random.choice([8, 14, 28])
            for i in range(0, IMG_SIZE, w):
                fill = c1 if (i // w) % 2 == 0 else c2
                draw.rectangle([i, 0, i + w, IMG_SIZE], fill=fill)

    def draw_shape(draw, shape, bbox, color):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if shape == "circle":
            draw.ellipse(bbox, fill=color)
        elif shape == "rectangle":
            draw.rectangle(bbox, fill=color)
        elif shape == "ellipse":
            draw.ellipse([x1, y1 + (y2 - y1) // 4, x2, y2 - (y2 - y1) // 4], fill=color)
        elif shape == "triangle":
            draw.polygon([(cx, y1), (x1, y2), (x2, y2)], fill=color)
        elif shape == "diamond":
            draw.polygon([(cx, y1), (x2, cy), (cx, y2), (x1, cy)], fill=color)
        elif shape == "star":
            # Simple 5-point star
            pts = []
            for i in range(5):
                angle = math.radians(i * 72 - 90)
                r = (x2 - x1) // 2
                pts.append((int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))))
                angle2 = math.radians(i * 72 - 90 + 36)
                r2 = r // 2
                pts.append((int(cx + r2 * math.cos(angle2)), int(cy + r2 * math.sin(angle2))))
            draw.polygon(pts, fill=color)
        elif shape == "line":
            draw.line([x1, y1, x2, y2], fill=color, width=max(3, (x2 - x1) // 8))

    manifest = []
    for i in range(count):
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        draw = ImageDraw.Draw(img)

        # Background
        bg_type = random.choice(bg_types)
        bg_c1 = random.choice(colors)
        bg_c2 = random.choice(colors)
        while bg_c2 == bg_c1:
            bg_c2 = random.choice(colors)
        make_background(draw, img, bg_type, bg_c1, bg_c2)

        # 1-3 shapes
        num_shapes = random.randint(1, 3)
        caption_parts = []

        if bg_type == "solid":
            caption_parts.append(f"a {color_names[bg_c1]} background")
        elif bg_type in ("gradient_h", "gradient_v"):
            direction = "horizontal" if bg_type == "gradient_h" else "vertical"
            caption_parts.append(f"a {direction} gradient from {color_names[bg_c1]} to {color_names[bg_c2]}")
        elif bg_type == "checkerboard":
            caption_parts.append(f"a {color_names[bg_c1]} and {color_names[bg_c2]} checkerboard")
        elif bg_type == "stripes":
            caption_parts.append(f"{color_names[bg_c1]} and {color_names[bg_c2]} stripes")

        for _ in range(num_shapes):
            shape = random.choice(shape_types)
            color = random.choice(colors)
            size_idx = random.randint(0, 3)
            size_name = size_names[size_idx]
            base_sizes = [20, 35, 55, 80]
            sz = base_sizes[size_idx]

            # Position on 3x3 grid
            gx, gy = random.randint(0, 2), random.randint(0, 2)
            cx = int((gx + 0.5) * IMG_SIZE / 3)
            cy = int((gy + 0.5) * IMG_SIZE / 3)
            bbox = [cx - sz // 2, cy - sz // 2, cx + sz // 2, cy + sz // 2]
            bbox = [max(0, b) for b in bbox[:2]] + [min(IMG_SIZE, b) for b in bbox[2:]]

            draw_shape(draw, shape, bbox, color)
            pos_name = pos_names.get((gx, gy), "center")
            caption_parts.append(f"a {size_name} {color_names[color]} {shape} at {pos_name}")

        caption = "Image with " + ", ".join(caption_parts) + "."

        fname = f"img_{i:05d}.png"
        img.save(os.path.join(output_dir, fname))
        manifest.append({"image": fname, "caption": caption})

        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{count} images")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Written {count} images to {output_dir}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Sample: {manifest[0]['caption']}")


# ============================================================
# 3. AUDIO (ASR + TTS) GENERATOR
# ============================================================

def _init_tts_engine():
    """Initialize pyttsx3 TTS engine with 16kHz output."""
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speaking rate
    engine.setProperty('volume', 0.9)
    return engine


def generate_audio_data(asr_csv: str, tts_csv: str, audio_dir: str, texts: list, count: int = 5000):
    """Generate real speech audio using pyttsx3 (Windows SAPI5 voices)."""
    import pyttsx3
    print(f"Generating {count} audio samples with pyttsx3 TTS...")

    os.makedirs(audio_dir, exist_ok=True)
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    print(f"  Available voices: {len(voices)}")
    for v in voices:
        print(f"    {v.id.split(chr(92))[-1]}")

    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)

    # Use short sentences for audio (< 60 chars for reasonable duration)
    short_texts = [t for t in texts if 10 < len(t) < 60]
    if len(short_texts) < count:
        short_texts = short_texts * (count // len(short_texts) + 1)
    random.shuffle(short_texts)
    short_texts = short_texts[:count]

    asr_rows = []
    tts_rows = []

    for i, text in enumerate(short_texts):
        # Cycle through voices
        voice_idx = i % len(voices)
        engine.setProperty('voice', voices[voice_idx].id)

        # Vary speaking rate slightly for diversity
        rate = 130 + (i % 5) * 10  # 130-170 wpm
        engine.setProperty('rate', rate)

        wav_path = os.path.join(audio_dir, f"audio_{i:05d}.wav")
        engine.save_to_file(text, wav_path)

        asr_rows.append((wav_path, text))
        tts_rows.append((text, wav_path))

        # pyttsx3 queues commands — run in batches to avoid memory buildup
        if (i + 1) % 100 == 0:
            engine.runAndWait()
            print(f"  Generated {i + 1}/{count} audio files")

    # Flush remaining
    engine.runAndWait()
    print(f"  Generated {count}/{count} audio files")

    # Convert to 16kHz mono if needed (pyttsx3 may output 22050Hz)
    try:
        import subprocess
        first_wav = os.path.join(audio_dir, "audio_00000.wav")
        if os.path.exists(first_wav):
            with wave.open(first_wav, 'r') as wf:
                sr = wf.getframerate()
                if sr != 16000:
                    print(f"  Converting from {sr}Hz to 16000Hz...")
                    for i in range(count):
                        src = os.path.join(audio_dir, f"audio_{i:05d}.wav")
                        tmp = src + ".tmp"
                        if os.path.exists(src):
                            # Use ffmpeg if available
                            result = subprocess.run(
                                ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", tmp],
                                capture_output=True, timeout=10
                            )
                            if result.returncode == 0:
                                os.replace(tmp, src)
                            else:
                                # Skip conversion if ffmpeg not available
                                if os.path.exists(tmp):
                                    os.remove(tmp)
                                if i == 0:
                                    print(f"  ffmpeg not available, keeping original sample rate ({sr}Hz)")
                                break
                    else:
                        print(f"  Converted all {count} files to 16kHz mono")
    except Exception as e:
        print(f"  Note: Could not convert sample rate: {e}")

    # Write ASR CSV (wav,text)
    os.makedirs(os.path.dirname(asr_csv) or ".", exist_ok=True)
    with open(asr_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wav", "text"])
        writer.writerows(asr_rows)

    # Write TTS CSV (text,wav)
    os.makedirs(os.path.dirname(tts_csv) or ".", exist_ok=True)
    with open(tts_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "wav"])
        writer.writerows(tts_rows)

    print(f"  ASR CSV: {asr_csv} ({len(asr_rows)} rows)")
    print(f"  TTS CSV: {tts_csv} ({len(tts_rows)} rows)")


# ============================================================
# 4. OCR DATA GENERATOR
# ============================================================

def generate_ocr_data(csv_path: str, image_dir: str, count: int = 5000):
    """Generate OCR training images with diverse text rendering."""
    from PIL import Image, ImageDraw, ImageFont
    print(f"Generating {count} OCR samples...")

    os.makedirs(image_dir, exist_ok=True)

    # Text pool: words, numbers, mixed
    words = [
        "hello", "world", "python", "data", "model", "train", "test", "image",
        "audio", "vision", "learn", "neural", "deep", "code", "open", "fast",
        "smart", "cloud", "blue", "red", "green", "white", "black", "gold",
        "sun", "moon", "star", "tree", "lake", "hill", "road", "city",
        "book", "page", "word", "line", "font", "size", "bold", "text",
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "cat", "dog", "bird", "fish", "bear", "lion", "wolf", "deer",
    ]

    bg_colors = [(255, 255, 255), (240, 240, 240), (200, 200, 200),
                 (255, 255, 230), (230, 240, 255), (240, 255, 240),
                 (50, 50, 50), (30, 30, 60), (60, 30, 30)]

    text_colors = [(0, 0, 0), (50, 50, 50), (0, 0, 128), (128, 0, 0),
                   (0, 100, 0), (255, 255, 255), (200, 200, 200)]

    rows = []
    for i in range(count):
        # Generate text: 1-4 words or a number
        text_type = random.choice(["word", "words", "number", "mixed", "sentence"])
        if text_type == "word":
            text = random.choice(words)
        elif text_type == "words":
            text = " ".join(random.sample(words, random.randint(2, 4)))
        elif text_type == "number":
            text = str(random.randint(0, 99999))
        elif text_type == "mixed":
            text = f"{random.choice(words)} {random.randint(0, 999)}"
        else:
            text = f"{random.choice(words)} is {random.choice(words)}"

        # Randomly uppercase
        if random.random() < 0.3:
            text = text.upper()
        elif random.random() < 0.3:
            text = text.capitalize()

        # Image
        img_size = 224
        bg = random.choice(bg_colors)
        fg = random.choice(text_colors)
        # Ensure contrast
        while abs(sum(bg) - sum(fg)) < 200:
            fg = random.choice(text_colors)

        img = Image.new("RGB", (img_size, img_size), bg)
        draw = ImageDraw.Draw(img)

        font_size = random.randint(16, 48)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Center text with slight random offset
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (img_size - tw) // 2 + random.randint(-20, 20)
        y = (img_size - th) // 2 + random.randint(-20, 20)
        x = max(5, min(img_size - tw - 5, x))
        y = max(5, min(img_size - th - 5, y))

        draw.text((x, y), text, fill=fg, font=font)

        # Random noise
        if random.random() < 0.3:
            for _ in range(random.randint(10, 50)):
                nx, ny = random.randint(0, img_size - 1), random.randint(0, img_size - 1)
                img.putpixel((nx, ny), tuple(random.randint(0, 255) for _ in range(3)))

        fname = f"ocr_{i:05d}.png"
        img.save(os.path.join(image_dir, fname))
        rows.append((fname, text))

        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{count} OCR images")

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "text"])
        writer.writerows(rows)

    print(f"  OCR CSV: {csv_path} ({len(rows)} rows)")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--all", action="store_true", help="Generate all modalities")
    parser.add_argument("--text", action="store_true", help="Text corpus")
    parser.add_argument("--audio", action="store_true", help="ASR + TTS audio")
    parser.add_argument("--images", action="store_true", help="Images + captions")
    parser.add_argument("--ocr", action="store_true", help="OCR images + text")
    parser.add_argument("--count", type=int, default=5000, help="Samples per modality")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not any([args.all, args.text, args.audio, args.images, args.ocr]):
        args.all = True

    random.seed(args.seed)
    count = args.count

    # Generate text first (shared by audio)
    texts = []
    if args.all or args.text or args.audio:
        generate_text_corpus("data/text/production_corpus.txt", count)
        with open("data/text/production_corpus.txt", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

    if args.all or args.audio:
        if not texts:
            with open("data/text/production_corpus.txt", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
        generate_audio_data(
            "data/audio/production_asr.csv",
            "data/audio/production_tts.csv",
            "data/audio",
            texts,
            count
        )

    if args.all or args.images:
        generate_images("data/images", "data/images/production_annotations.json", count)

    if args.all or args.ocr:
        generate_ocr_data("data/ocr/production_ocr.csv", "data/ocr", count)

    print("\nDone! Generated data summary:")
    for p in ["data/text/production_corpus.txt", "data/audio/production_asr.csv",
              "data/audio/production_tts.csv", "data/images/production_annotations.json",
              "data/ocr/production_ocr.csv"]:
        if os.path.exists(p):
            if p.endswith(".txt"):
                with open(p, encoding="utf-8") as f:
                    n = sum(1 for _ in f)
            elif p.endswith(".csv"):
                with open(p, encoding="utf-8") as f:
                    n = sum(1 for _ in f) - 1
            elif p.endswith(".json"):
                n = len(json.load(open(p, encoding="utf-8")))
            else:
                n = "?"
            print(f"  {p}: {n} samples")


if __name__ == "__main__":
    main()
