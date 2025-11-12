
import os, csv, random, json, wave, struct, math
from PIL import Image, ImageDraw, ImageFont

base = 'data'
os.makedirs(base, exist_ok=True)

# Text
os.makedirs(f'{base}/text', exist_ok=True)
random.seed(0)
with open(f'{base}/text/tiny_corpus.txt', 'w', encoding='utf-8') as f:
    for i in range(5000):
        f.write(f"This is a tiny training sentence number {i}. It describes a red square on a blue background.\n")

# Images (generate geometric shapes + captions)
img_root = f'{base}/images/images'
os.makedirs(img_root, exist_ok=True)
ann = []
for i in range(500):
    W=H=224
    img = Image.new('RGB', (W,H), (0,0,255))
    draw = ImageDraw.Draw(img)
    size = random.randint(40,120)
    x = random.randint(0, W-size)
    y = random.randint(0, H-size)
    draw.rectangle([x,y, x+size, y+size], fill=(255,0,0))
    path = f'{img_root}/{i:06d}.png'
    img.save(path)
    ann.append({"image": f"{i:06d}.png", "caption": f"A red square roughly {size}px on a blue background."})

with open(f'{base}/images/annotations.json', 'w', encoding='utf-8') as f:
    json.dump(ann, f)

# Audio (synthetic beeps + transcripts)
aud_dir = f'{base}/audio/wav'
os.makedirs(aud_dir, exist_ok=True)
asr_rows = []
tts_rows = []
sr = 16000
for i in range(300):
    dur = 1.0 + random.random()
    n = int(sr*dur)
    freq = 300 + random.randint(0,200)
    wav_path = f"{aud_dir}/{i:06d}.wav"
    with wave.open(wav_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for t in range(n):
            val = int(32767*0.2*math.sin(2*math.pi*freq*t/sr))
            wf.writeframes(struct.pack('<h', val))
    text = f"synthetic tone at {freq} hertz number {i}"
    asr_rows.append([wav_path, text])
    tts_rows.append([text, wav_path])

with open(f'{base}/audio/asr.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(["wav","text"]); w.writerows(asr_rows)
with open(f'{base}/audio/tts.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(["text","wav"]); w.writerows(tts_rows)

print("Synthetic datasets created under data/.")
