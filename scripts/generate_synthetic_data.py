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
import time
import multiprocessing as mp

# ============================================================
# 1. TEXT CORPUS GENERATOR
# ============================================================

def generate_text_corpus(output_path: str, count: int = 5000):
    """Generate diverse text: 40% math/numeric, 30% knowledge facts, 30% grammar sentences."""
    print(f"Generating {count} text samples (math + knowledge + grammar)...")

    # ============ MATH & NUMERIC (40%) ============

    def gen_addition():
        a, b = random.randint(0, 100), random.randint(0, 100)
        return f"{a} + {b} = {a + b}."

    def gen_subtraction():
        a = random.randint(0, 100); b = random.randint(0, a)
        return f"{a} - {b} = {a - b}."

    def gen_multiplication():
        a, b = random.randint(0, 12), random.randint(0, 12)
        return f"{a} x {b} = {a * b}."

    def gen_division():
        b = random.randint(1, 12); r = random.randint(0, 12); a = b * r
        return f"{a} / {b} = {r}."

    def gen_word_math():
        a, b = random.randint(1, 50), random.randint(1, 50)
        return random.choice([
            f"{a} plus {b} equals {a + b}.",
            f"{a} minus {b} equals {abs(a - b)}.",
            f"{a} times {b} equals {a * b}.",
        ])

    def gen_counting():
        start = random.randint(0, 50)
        step = random.choice([1, 2, 3, 5, 10])
        nums = [str(start + i * step) for i in range(random.randint(4, 8))]
        return f"Count by {step}: {', '.join(nums)}."

    def gen_comparison_num():
        a, b = random.randint(1, 100), random.randint(1, 100)
        if a > b: return f"{a} is greater than {b}."
        elif a < b: return f"{a} is less than {b}."
        else: return f"{a} is equal to {b}."

    def gen_even_odd():
        n = random.randint(1, 100)
        return f"{n} is {'even' if n % 2 == 0 else 'odd'}."

    def gen_ordinal():
        ords = {1:"first",2:"second",3:"third",4:"fourth",5:"fifth",6:"sixth",
                7:"seventh",8:"eighth",9:"ninth",10:"tenth",11:"eleventh",12:"twelfth"}
        n = random.randint(1, 12)
        return f"The {ords[n]} number is {n}."

    def gen_word_problem():
        name = random.choice(["Alice","Bob","Sam","Emma","Tom","Mary","Liam","Zara"])
        item = random.choice(["apples","books","coins","eggs","pencils","stickers"])
        a, b = random.randint(3, 20), random.randint(1, 10)
        return random.choice([
            f"{name} has {a} {item} and gets {b} more. Now {name} has {a+b} {item}.",
            f"{name} has {a} {item} and gives {min(b,a)} away. Now {name} has {a-min(b,a)} {item}.",
            f"Each bag has {a} {item}. There are {b} bags. That is {a*b} {item} in total.",
        ])

    def gen_fraction():
        n, d = random.randint(1, 9), random.randint(2, 10)
        halves = {2:"half",3:"third",4:"quarter",5:"fifth",6:"sixth",7:"seventh",8:"eighth",9:"ninth",10:"tenth"}
        return f"{n}/{d} means {n} {halves.get(d,f'{d}th')}{'s' if n>1 else ''}."

    def gen_percentage():
        p, w = random.randint(1, 100), random.choice([10,20,50,100])
        return f"{p} out of {w} is {round(p/w*100,1)} percent."

    def gen_sequence():
        return random.choice([
            f"Square numbers: {', '.join(str(i*i) for i in range(1,7))}.",
            f"Powers of 2: {', '.join(str(2**i) for i in range(8))}.",
            f"Fibonacci: 1, 1, 2, 3, 5, 8, 13, 21.",
            f"Triangular numbers: 1, 3, 6, 10, 15, 21.",
            f"Prime numbers: 2, 3, 5, 7, 11, 13, 17, 19.",
        ])

    def gen_geometry():
        return random.choice([
            f"A triangle has 3 sides and 3 angles. The angles add up to 180 degrees.",
            f"A square has 4 equal sides. Its area is side times side.",
            f"A rectangle with length {(l:=random.randint(2,10))} and width {(w:=random.randint(2,10))} has area {l*w}.",
            f"A circle with radius {(r:=random.randint(1,10))} has diameter {2*r}.",
            f"The perimeter of a square with side {(s:=random.randint(1,10))} is {4*s}.",
            f"A pentagon has 5 sides. A hexagon has 6 sides. An octagon has 8 sides.",
        ])

    def gen_time_math():
        h, m = random.randint(1,12), random.choice([0,15,30,45])
        ah = random.randint(1,5)
        return f"It is {h}:{m:02d}. In {ah} hours it will be {(h+ah-1)%12+1}:{m:02d}."

    # ============ KNOWLEDGE & FACTS (30%) ============

    def gen_science():
        return random.choice([
            "Water boils at 100 degrees Celsius.", "Ice melts at 0 degrees Celsius.",
            "The sun is a star at the center of our solar system.", "The moon orbits the earth.",
            "Light travels faster than sound.", "Sound travels through air as waves.",
            "Plants make food using sunlight. This is called photosynthesis.",
            "The earth rotates once every 24 hours.", "The earth orbits the sun once a year.",
            "Gravity pulls objects toward the ground.", "Magnets attract iron and steel.",
            "Electricity flows through conductors like copper wire.",
            "Atoms are the building blocks of matter.", "Water is made of hydrogen and oxygen.",
            "The human body has 206 bones.", "The heart pumps blood through the body.",
            "The brain controls thinking and movement.", "Muscles help the body move.",
            "Trees produce oxygen and absorb carbon dioxide.",
            "Bees pollinate flowers and make honey.", "Spiders have 8 legs. Insects have 6.",
        ])

    def gen_geography():
        return random.choice([
            "The earth has 7 continents and 5 oceans.", "Asia is the largest continent.",
            "The Pacific Ocean is the largest ocean.", "Mount Everest is the tallest mountain.",
            "The Nile is one of the longest rivers in the world.",
            "The Amazon rainforest is the largest tropical forest.",
            "Deserts are dry places with very little rain.", "Islands are land surrounded by water.",
            "Volcanoes can erupt and release hot lava.", "Glaciers are large bodies of moving ice.",
            "The North Pole is at the top of the earth. The South Pole is at the bottom.",
            "Rivers flow from mountains to the sea.", "Lakes are bodies of water surrounded by land.",
        ])

    def gen_definition():
        return random.choice([
            "A book is a collection of written pages bound together.",
            "A teacher is a person who helps students learn new things.",
            "A doctor is a person who treats people when they are sick.",
            "A school is a place where children go to learn.",
            "A library is a building where people can borrow books.",
            "A hospital is a place where sick people receive care.",
            "A farm is land where crops are grown and animals are raised.",
            "A bridge is a structure built to cross over water or a valley.",
            "A clock is a device used to measure and show the time.",
            "A map is a drawing that shows where places are located.",
            "A garden is a piece of ground where flowers and plants grow.",
            "An island is a piece of land completely surrounded by water.",
        ])

    def gen_cause_effect():
        c, e = random.choice([
            ("it rains", "the ground gets wet"), ("the sun sets", "the sky becomes dark"),
            ("you study hard", "you learn more"), ("you eat healthy food", "your body stays strong"),
            ("you exercise daily", "you become more fit"), ("ice is heated", "it melts into water"),
            ("you plant a seed and water it", "a plant begins to grow"),
            ("the wind blows hard", "the trees sway back and forth"),
            ("you turn off the light", "the room becomes dark"),
            ("winter comes", "the temperature drops and it gets cold"),
        ])
        return random.choice([f"When {c}, {e}.", f"If {c}, then {e}.", f"Because {c}, {e}."])

    def gen_qa():
        q, a = random.choice([
            ("What color is the sky?", "The sky is blue."),
            ("How many legs does a dog have?", "A dog has four legs."),
            ("How many days are in a week?", "There are seven days in a week."),
            ("How many months are in a year?", "There are twelve months in a year."),
            ("What do plants need to grow?", "Plants need water, sunlight, and soil."),
            ("How many sides does a triangle have?", "A triangle has three sides."),
            ("What is the opposite of hot?", "The opposite of hot is cold."),
            ("Where does the sun rise?", "The sun rises in the east."),
            ("What season comes after winter?", "Spring comes after winter."),
            ("What is the largest planet?", "Jupiter is the largest planet in our solar system."),
            ("How many hours are in a day?", "There are twenty-four hours in a day."),
            ("What is the boiling point of water?", "Water boils at one hundred degrees Celsius."),
        ])
        return f"{q} {a}"

    # ============ GRAMMAR & SENTENCES (30%) ============

    subjects_s = [  # singular (verb takes 's')
        "the cat", "a dog", "the bird", "the teacher", "a student", "the doctor",
        "the farmer", "the scientist", "the chef", "the dancer", "the old man",
        "a young woman", "the little boy", "the wise owl", "a brave knight",
    ]
    subjects_p = [  # plural (verb takes base form)
        "the cats", "two dogs", "the birds", "the teachers", "some students",
        "the children", "many farmers", "the scientists", "three friends",
    ]
    # (base_form, third_person_s)
    verbs_i = [
        ("run","runs"), ("walk","walks"), ("sleep","sleeps"), ("sing","sings"),
        ("dance","dances"), ("jump","jumps"), ("swim","swims"), ("fly","flies"),
        ("laugh","laughs"), ("smile","smiles"), ("think","thinks"), ("read","reads"),
        ("play","plays"), ("work","works"), ("dream","dreams"), ("travel","travels"),
    ]
    verbs_t = [
        ("see","sees"), ("like","likes"), ("love","loves"), ("find","finds"),
        ("make","makes"), ("build","builds"), ("draw","draws"), ("cook","cooks"),
        ("eat","eats"), ("carry","carries"), ("hold","holds"), ("open","opens"),
    ]
    objects = [
        "a book", "the ball", "some food", "a letter", "the door", "a picture",
        "the cake", "a flower", "the key", "a map", "the house", "a garden",
    ]
    locations = [
        "in the park", "at home", "near the river", "on the hill", "by the lake",
        "at school", "in the city", "on the farm", "at the library", "in the garden",
    ]
    adjectives = [
        "big", "small", "red", "blue", "green", "old", "new", "fast", "slow",
        "bright", "warm", "cold", "soft", "tall", "short", "heavy", "light",
    ]
    adverbs = ["quickly","slowly","carefully","happily","quietly","gently","always","often"]

    def gen_sv():
        if random.random() < 0.5:
            s = random.choice(subjects_s); _, v = random.choice(verbs_i)
        else:
            s = random.choice(subjects_p); v, _ = random.choice(verbs_i)
        return f"{s.capitalize()} {v} {random.choice(locations)}."

    def gen_svo():
        if random.random() < 0.5:
            s = random.choice(subjects_s); _, v = random.choice(verbs_t)
        else:
            s = random.choice(subjects_p); v, _ = random.choice(verbs_t)
        return f"{s.capitalize()} {v} {random.choice(objects)}."

    def gen_adverb_sent():
        s = random.choice(subjects_s); _, v = random.choice(verbs_i)
        return f"{s.capitalize()} {random.choice(adverbs)} {v} {random.choice(locations)}."

    def gen_compound_sent():
        s1 = random.choice(subjects_s); _, v1 = random.choice(verbs_i)
        s2 = random.choice(subjects_s); _, v2 = random.choice(verbs_i)
        conj = random.choice(["and", "but", "so", "while"])
        return f"{s1.capitalize()} {v1} {conj} {s2} {v2}."

    def gen_description():
        s = random.choice(subjects_s)
        a1, a2 = random.sample(adjectives, 2)
        return f"{s.capitalize()} is {a1} and {a2}."

    def gen_possessive():
        owner = random.choice(["My","Your","His","Her","Their","Our"])
        item = random.choice(["house","car","book","garden","idea","plan","cat","dog"])
        adj = random.choice(adjectives)
        return f"{owner} {item} is very {adj}."

    def gen_there_is():
        n = random.randint(2, 10)
        thing = random.choice(["cats","birds","trees","houses","books","flowers","stars"])
        loc = random.choice(locations)
        return f"There are {n} {thing} {loc}."

    def gen_list_sent():
        items = random.sample(["apples","bread","milk","eggs","rice","cheese","tea","soup","fish","cake"], random.randint(3,5))
        return f"I need {', '.join(items[:-1])} and {items[-1]}."

    # ============ WEIGHTED GENERATOR POOLS ============

    math_gens = [
        gen_addition, gen_subtraction, gen_multiplication, gen_division,
        gen_word_math, gen_counting, gen_comparison_num, gen_even_odd,
        gen_ordinal, gen_word_problem, gen_fraction, gen_percentage,
        gen_sequence, gen_geometry, gen_time_math,
    ]

    knowledge_gens = [
        gen_science, gen_geography, gen_definition, gen_cause_effect, gen_qa,
    ]

    grammar_gens = [
        gen_sv, gen_svo, gen_adverb_sent, gen_compound_sent,
        gen_description, gen_possessive, gen_there_is, gen_list_sent,
    ]

    # Weighted: 40% math, 30% knowledge, 30% grammar
    def pick_generator():
        r = random.random()
        if r < 0.4: return random.choice(math_gens)
        elif r < 0.7: return random.choice(knowledge_gens)
        else: return random.choice(grammar_gens)

    generators = None  # Not used — we use pick_generator()

    sentences = set()
    while len(sentences) < count:
        gen = pick_generator()
        sent = gen()
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


def _tts_synthesize_one(
    i: int,
    text: str,
    wav_path: str,
    voice_ids: list[str],
    volume: float,
    base_rate: int,
    rate_step: int,
):
    """Worker process: synthesize exactly one file via pyttsx3/SAPI5."""
    import pyttsx3

    # Skip if already generated (supports resume)
    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
        return

    engine = pyttsx3.init()
    engine.setProperty("volume", volume)

    voice_id = voice_ids[i % len(voice_ids)]
    engine.setProperty("voice", voice_id)

    rate = base_rate + (i % 5) * rate_step
    engine.setProperty("rate", rate)

    engine.save_to_file(text, wav_path)
    engine.runAndWait()

    try:
        engine.stop()
    except Exception:
        pass


def generate_audio_data(
    asr_csv: str,
    tts_csv: str,
    audio_dir: str,
    texts: list,
    count: int = 5000,
    *,
    per_file_timeout_sec: int = 30,
):
    """Generate speech audio using pyttsx3 (Windows SAPI5 voices), with hang-safe batching."""
    import pyttsx3
    print(f"Generating {count} audio samples with pyttsx3 TTS...")

    os.makedirs(audio_dir, exist_ok=True)

    # Probe voices in main process (fast + reliable)
    probe = pyttsx3.init()
    voices = probe.getProperty("voices") or []
    voice_ids = [v.id for v in voices]
    print(f"  Available voices: {len(voice_ids)}")
    for vid in voice_ids:
        try:
            print(f"    {vid.split(chr(92))[-1]}")
        except Exception:
            print(f"    {vid}")
    try:
        probe.stop()
    except Exception:
        pass

    if not voice_ids:
        raise RuntimeError("pyttsx3 returned no voices; cannot generate audio on this system.")

    # Use short sentences for audio (< 60 chars for reasonable duration)
    short_texts = [t for t in texts if 10 < len(t) < 60]
    if not short_texts:
        raise RuntimeError("No short texts (10<len<60) available for audio generation.")
    if len(short_texts) < count:
        short_texts = short_texts * (count // len(short_texts) + 1)
    random.shuffle(short_texts)
    short_texts = short_texts[:count]

    # Prepare deterministic paths
    items: list[tuple[int, str, str]] = []
    for i, text in enumerate(short_texts):
        wav_path = os.path.join(audio_dir, f"audio_{i:05d}.wav")
        items.append((i, text, wav_path))

    # Synthesize file-by-file in isolated processes so we can terminate on hangs
    total = len(items)
    start_t = time.time()

    # Ensure spawn on Windows (safe with pyttsx3/COM)
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    last_report_done = -1
    hung = 0
    for i, text, wav_path in items:
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            continue

        p = mp.Process(
            target=_tts_synthesize_one,
            args=(i, text, wav_path, voice_ids, 0.9, 130, 10),
            daemon=True,
        )
        p.start()
        p.join(timeout=per_file_timeout_sec)

        if p.is_alive():
            hung += 1
            p.terminate()
            p.join(timeout=5)
            # Leave a placeholder empty file only if it doesn't exist; helps spot gaps
            try:
                if not os.path.exists(wav_path):
                    Path(wav_path).touch()
            except Exception:
                pass
            if hung <= 5 or hung % 25 == 0:
                print(
                    f"  Warning: TTS hung on index {i} (killed after {per_file_timeout_sec}s). "
                    f"Hung so far: {hung}"
                )

        # Live progress after each attempt (cheap scan)
        done = 0
        for _, _, wp in items:
            if os.path.exists(wp) and os.path.getsize(wp) > 0:
                done += 1
        if done != last_report_done:
            elapsed = int(time.time() - start_t)
            print(f"  Generated {done}/{total} audio files (elapsed {elapsed}s)")
            last_report_done = done

    # Final progress
    done = 0
    for _, _, wav_path in items:
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            done += 1
    elapsed = int(time.time() - start_t)
    print(f"  Generated {done}/{total} audio files (elapsed {elapsed}s)")
    if hung:
        print(f"  Note: {hung} items hung and were skipped; rerun to retry those indices.")

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
        # Write rows for files that exist (supports partial runs/resume)
        for i, text, wav_path in items:
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                writer.writerow([wav_path, text])

    # Write TTS CSV (text,wav)
    os.makedirs(os.path.dirname(tts_csv) or ".", exist_ok=True)
    with open(tts_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "wav"])
        for i, text, wav_path in items:
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                writer.writerow([text, wav_path])

    # Report counts from CSVs (excluding header)
    try:
        with open(asr_csv, encoding="utf-8") as f:
            asr_n = sum(1 for _ in f) - 1
    except Exception:
        asr_n = "?"
    try:
        with open(tts_csv, encoding="utf-8") as f:
            tts_n = sum(1 for _ in f) - 1
    except Exception:
        tts_n = "?"
    print(f"  ASR CSV: {asr_csv} ({asr_n} rows)")
    print(f"  TTS CSV: {tts_csv} ({tts_n} rows)")


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
            "data/audio/wav",
            texts,
            count
        )

    if args.all or args.images:
        generate_images("data/images/images", "data/images/production_annotations.json", count)

    if args.all or args.ocr:
        generate_ocr_data("data/ocr/production_ocr.csv", "data/ocr/images", count)

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
