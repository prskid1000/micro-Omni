
# This script provides *examples* for sub-5GB datasets.
# Uncomment and adjust URLs/paths for your environment.
import argparse, os, subprocess, json, random, csv

def ensure(p): os.makedirs(p, exist_ok=True)

def dl_text():
    ensure("data/text")
    # Example: Wikitext-2 small
    # subprocess.check_call(["wget", "-c", "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip", "-O", "wikitext-2-v1.zip"])
    # subprocess.check_call(["unzip", "-o", "wikitext-2-v1.zip", "-d", "data/text/"])
    print("Please download a small text corpus (e.g., Wikitext-2) and place under data/text/.")

def dl_images():
    ensure("data/images/images")
    # Example: COCO 2017 val (5k images, ~1GB) + captions (~250MB)
    # subprocess.check_call(["wget","-c","http://images.cocodataset.org/zips/val2017.zip","-O","val2017.zip"])
    # subprocess.check_call(["unzip","-o","val2017.zip","-d","data/images/"])
    # subprocess.check_call(["mv","data/images/val2017","data/images/images"])
    # subprocess.check_call(["wget","-c","http://images.cocodataset.org/annotations/annotations_trainval2017.zip","-O","ann.zip"])
    # subprocess.check_call(["unzip","-o","ann.zip","-d","data/images/"])
    # Create a small annotations.json mapping (image, caption) from captions_val2017.json
    print("Please download COCO val2017 images and captions, then convert to data/images/annotations.json.")

def dl_audio_asr():
    ensure("data/audio/wav")
    # Example: Common Voice EN subset (keep first ~20 hours)
    print("Please download a small ASR dataset (e.g., Common Voice subset) and write a CSV data/audio/asr.csv with [wav, text].")

def dl_audio_tts():
    ensure("data/audio/wav")
    # Example: LJSpeech subset (first 3k clips)
    print("Please download a small TTS dataset (e.g., LJSpeech subset) and write data/audio/tts.csv with [text, wav].")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", nargs="+", required=True, choices=["text","images","audio_asr","audio_tts"])
    args = ap.parse_args()
    for m in args.modality:
        globals()[f"dl_{m}"]()
    print("Done (create under-5GB subsets before running training).")
