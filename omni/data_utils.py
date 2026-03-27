import csv
import json
import os
import random
from itertools import zip_longest

import torch
import torchaudio
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision import transforms

from omni.io_utils import load_audio


def collate_mel_fn(batch, max_mel_length=None):
    n_mels = batch[0].shape[1]
    max_len = max_mel_length if max_mel_length is not None else max(m.shape[0] for m in batch)
    padded = []
    mel_lengths = []
    for m in batch:
        current_len = m.shape[0]
        pad_len = max_len - current_len
        if pad_len > 0:
            pad = m.new_zeros(pad_len, n_mels)
            m = torch.cat([m, pad], dim=0)
        padded.append(m)
        mel_lengths.append(current_len)
    return torch.stack(padded), torch.tensor(mel_lengths, dtype=torch.long)


def collate_mel_text_fn(batch, max_mel_length=None):
    mels, texts = zip(*batch)
    n_mels = mels[0].shape[1]
    max_len = max_mel_length if max_mel_length is not None else max(m.shape[0] for m in mels)
    padded_mels = []
    mel_lengths = []
    for m in mels:
        current_len = m.shape[0]
        pad_len = max_len - current_len
        if pad_len > 0:
            pad = m.new_zeros(pad_len, n_mels)
            m = torch.cat([m, pad], dim=0)
        padded_mels.append(m)
        mel_lengths.append(current_len)
    return torch.stack(padded_mels), list(texts), torch.tensor(mel_lengths, dtype=torch.long)


def collate_mel_audio_fn(batch, max_mel_length=None, max_audio_length=None):
    mels, audios = zip(*batch)
    n_mels = mels[0].shape[1]
    max_mel_len = max_mel_length if max_mel_length is not None else max(m.shape[0] for m in mels)
    padded_mels = []
    mel_lengths = []
    for m in mels:
        current_len = m.shape[0]
        pad_len = max_mel_len - current_len
        if pad_len > 0:
            pad = m.new_zeros(pad_len, n_mels)
            m = torch.cat([m, pad], dim=0)
        padded_mels.append(m)
        mel_lengths.append(current_len)
    max_audio_len = max_audio_length if max_audio_length is not None else max(a.shape[0] for a in audios)
    padded_audios = []
    audio_lengths = []
    for a in audios:
        current_len = a.shape[0]
        pad_len = max_audio_len - current_len
        if pad_len > 0:
            a = torch.cat([a, torch.zeros(pad_len, dtype=a.dtype, device=a.device)], dim=0)
        padded_audios.append(a)
        audio_lengths.append(current_len)
    return (
        torch.stack(padded_mels),
        torch.stack(padded_audios),
        torch.tensor(mel_lengths, dtype=torch.long),
        torch.tensor(audio_lengths, dtype=torch.long),
    )


class TextDataset(IterableDataset):
    def __init__(self, path, tokenizer, ctx, shuffle_buffer_size=10000, seed=None, skip_samples=0, filter_outliers=True, use_sentences=True):
        self.path, self.tok, self.ctx = path, tokenizer, ctx
        self.shuffle_buffer_size, self.seed, self.skip_samples = shuffle_buffer_size, seed, skip_samples
        self.filter_outliers = filter_outliers
        self.use_sentences = use_sentences
        self._num_lines = None
        self._error_counts = {"exceeds_max_len": 0}

    def _split_into_sentences(self, text):
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def get_length(self):
        if self._num_lines is None:
            self._num_lines = sum(1 for _ in open(self.path, "r", encoding="utf-8", errors="ignore"))
        return self._num_lines

    def get_error_stats(self):
        return self._error_counts.copy()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        rng = random.Random(self.seed + worker_id if self.seed else None)
        buffer = []
        val_split = getattr(self, "_val_split", None)
        val_mode = getattr(self, "_val_mode", False)
        worker_skip = (self.skip_samples // num_workers) + (1 if worker_id < (self.skip_samples % num_workers) else 0)
        skipped = 0
        with open(self.path, "r", encoding="utf-8", errors="ignore", buffering=8192 * 1024) as f:
            for idx, line in enumerate(f):
                if idx % num_workers != worker_id:
                    continue
                if val_split and (hash((idx, self.seed)) % 100 < val_split * 100) != val_mode:
                    continue
                text = line.strip()
                if not text:
                    continue
                if skipped < worker_skip:
                    skipped += 1
                    continue
                sentences = self._split_into_sentences(text) if self.use_sentences else [text]
                for sentence in sentences:
                    if not sentence:
                        continue
                    ids = [1] + self.tok.encode(sentence)
                    if self.filter_outliers and len(ids) >= self.ctx:
                        self._error_counts["exceeds_max_len"] += 1
                        continue
                    x = torch.tensor(ids + [0] * (self.ctx - len(ids)), dtype=torch.long)
                    y = torch.cat([x[1:], torch.tensor([0], dtype=torch.long)])
                    if self.shuffle_buffer_size > 0:
                        buffer.append((x, y))
                        if len(buffer) >= self.shuffle_buffer_size:
                            yield buffer.pop(rng.randint(0, len(buffer) - 1))
                    else:
                        yield x, y
        if buffer:
            rng.shuffle(buffer)
            yield from buffer
        if self.skip_samples > 0:
            print(f"Dataset exhausted: resetting skip_samples from {self.skip_samples} to 0 for next epoch")
            self.skip_samples = 0


def build_char_vocab_from_asr_csv(csv_path):
    chars = set()
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            if text:
                chars.update(text)
    char_to_idx = {"<BLANK>": 0, "<UNK>": 1}
    idx_to_char = {0: "<BLANK>", 1: "<UNK>"}
    for char in sorted(chars):
        if char not in char_to_idx:
            idx = len(char_to_idx)
            char_to_idx[char] = idx
            idx_to_char[idx] = char
    vocab_size = len(char_to_idx)
    return char_to_idx, idx_to_char, vocab_size


def calculate_max_text_len_from_asr_csv(csv_path):
    max_len = 0
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            if text:
                max_len = max(max_len, len(text))
    return max_len


def calculate_max_mel_length_from_asr_csv(csv_path, sr=16000, n_mels=128, sample_size=None):
    import numpy as np

    melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=160, win_length=400, n_mels=n_mels)
    max_len = 0
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if sample_size and sample_size < len(rows):
            rows = random.sample(rows, sample_size)
        total = len(rows)
        for idx, row in enumerate(rows):
            if (idx + 1) % 100 == 0:
                print(f"  Calculating mel lengths: {idx + 1}/{total} files...", end="\r")
            try:
                wav_path = row.get("wav", "").strip()
                if not wav_path or not os.path.exists(wav_path):
                    continue
                wav, file_sr = load_audio(wav_path)
                if file_sr != sr:
                    wav = torchaudio.transforms.Resample(file_sr, sr)(wav)
                mel = melspec(wav)[0].T
                max_len = max(max_len, mel.shape[0])
            except Exception:
                continue
        if total > 0:
            print(f"  Calculated mel lengths: {total} files processed")
    if max_len > 0:
        max_len = int(np.ceil(max_len / 256) * 256)
    return max_len


def analyze_asr_dataset(csv_path, sr=16000, n_mels=128, sample_size=None, text_percentile=95.0, mel_percentile=95.0):
    import numpy as np

    chars = set()
    text_lengths = []
    mel_lengths = []
    melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=160, win_length=400, n_mels=n_mels)
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        total = len(rows)
        mel_indices = set(range(total))
        if sample_size and sample_size < total:
            mel_indices = set(random.sample(range(total), sample_size))
        for idx, row in enumerate(rows):
            if (idx + 1) % 100 == 0:
                print(f"  Analyzing dataset: {idx + 1}/{total} files...", end="\r")
            text = row.get("text", "").strip()
            if text:
                chars.update(text)
                text_lengths.append(len(text))
            if idx in mel_indices:
                try:
                    wav_path = row.get("wav", "").strip()
                    if wav_path and os.path.exists(wav_path):
                        wav, file_sr = load_audio(wav_path)
                        if file_sr != sr:
                            wav = torchaudio.transforms.Resample(file_sr, sr)(wav)
                        mel = melspec(wav)[0].T
                        mel_lengths.append(mel.shape[0])
                except Exception:
                    continue
        if total > 0:
            print(f"  Analyzed dataset: {total} files processed")
    char_to_idx = {"<BLANK>": 0, "<UNK>": 1}
    idx_to_char = {0: "<BLANK>", 1: "<UNK>"}
    for char in sorted(chars):
        if char not in char_to_idx:
            idx = len(char_to_idx)
            char_to_idx[char] = idx
            idx_to_char[idx] = char
    vocab_size = len(char_to_idx)
    max_text_len = int(np.ceil(np.percentile(np.array(text_lengths), text_percentile))) if text_lengths else 0
    if mel_lengths:
        max_mel_len = np.percentile(np.array(mel_lengths), mel_percentile)
        max_mel_len = int(np.ceil(max_mel_len / 256) * 256)
    else:
        max_mel_len = 0
    return char_to_idx, idx_to_char, vocab_size, max_text_len, max_mel_len


def analyze_tts_dataset(csv_path, sr=16000, n_mels=128, frame_ms=80, sample_size=None, mel_percentile=95.0):
    import numpy as np

    mel_lengths = []
    hop_length = int(sr * frame_ms / 1000)
    win_length = min(1024, hop_length * 4)
    melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=hop_length, win_length=win_length, n_mels=n_mels)
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        total = len(rows)
        mel_indices = set(range(total))
        if sample_size and sample_size < total:
            mel_indices = set(random.sample(range(total), sample_size))
        for idx, row in enumerate(rows):
            if (idx + 1) % 100 == 0:
                print(f"  Analyzing TTS dataset: {idx + 1}/{total} files...", end="\r")
            if idx in mel_indices:
                try:
                    wav_path = row.get("wav", "").strip()
                    if wav_path and os.path.exists(wav_path):
                        wav, file_sr = load_audio(wav_path)
                        if file_sr != sr:
                            wav = torchaudio.transforms.Resample(file_sr, sr)(wav)
                        mel = melspec(wav)[0].T
                        mel_lengths.append(mel.shape[0])
                except Exception:
                    continue
        if total > 0:
            print(f"  Analyzed TTS dataset: {total} files processed")
    if mel_lengths:
        max_mel_len = np.percentile(np.array(mel_lengths), mel_percentile)
        max_mel_len = int(np.ceil(max_mel_len / 256) * 256)
    else:
        max_mel_len = 0
    return max_mel_len


def analyze_ocr_dataset(csv_path, text_percentile=95.0):
    import numpy as np

    chars = set()
    text_lengths = []
    idx = -1
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if (idx + 1) % 1000 == 0:
                print(f"  Analyzing OCR dataset: {idx + 1} rows...", end="\r")
            text = row.get("text", "") or row.get("label", "") or row.get("text_label", "")
            if text:
                chars.update(text)
                text_lengths.append(len(text))
        total = idx + 1 if idx >= 0 else 0
        if total > 0:
            print(f"  Analyzed OCR dataset: {total} rows processed")
    char_to_idx = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
    idx_to_char = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}
    for char in sorted(chars):
        if char not in char_to_idx:
            i = len(char_to_idx)
            char_to_idx[char] = i
            idx_to_char[i] = char
    vocab_size = len(char_to_idx)
    max_text_len = int(np.ceil(np.percentile(np.array(text_lengths), text_percentile))) if text_lengths else 0
    return char_to_idx, idx_to_char, vocab_size, max_text_len


def analyze_text_dataset(text_path, tokenizer, sample_size=None, ctx_percentile=95.0, use_sentences=True):
    import numpy as np
    import re

    print(f"\n📊 Analyzing text dataset: {text_path}")
    token_lengths = []

    def split_sentences(text):
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        total_lines = len(lines)
        if sample_size and sample_size < total_lines:
            lines = random.sample(lines, sample_size)
        for idx, line in enumerate(lines):
            if (idx + 1) % 1000 == 0:
                print(f"  Analyzed {idx + 1}/{len(lines)} lines...", end="\r")
            text = line.strip()
            if not text:
                continue
            text_units = split_sentences(text) if use_sentences else [text]
            for unit in text_units:
                if unit:
                    tokens = tokenizer.encode(unit)
                    token_lengths.append(len(tokens) + 1)
    if not token_lengths:
        return 256
    token_lengths_arr = np.array(token_lengths)
    percentile_len = np.percentile(token_lengths_arr, ctx_percentile)
    ctx_len = int(np.ceil(percentile_len / 64) * 64)
    return ctx_len


def analyze_vocoder_dataset(csv_path, sr=16000, n_fft=1024, hop_length=256, n_mels=128, sample_size=None, audio_percentile=95.0):
    import numpy as np

    audio_lengths = []
    mel_lengths = []
    melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, n_mels=n_mels)
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        total = len(rows)
        audio_indices = set(range(total))
        if sample_size and sample_size < total:
            audio_indices = set(random.sample(range(total), sample_size))
        for idx, row in enumerate(rows):
            if (idx + 1) % 100 == 0:
                print(f"  Analyzing vocoder dataset: {idx + 1}/{total} files...", end="\r")
            if idx in audio_indices:
                try:
                    wav_path = row.get("wav", "").strip()
                    if wav_path and os.path.exists(wav_path):
                        wav, file_sr = load_audio(wav_path)
                        if file_sr != sr:
                            wav = torchaudio.transforms.Resample(file_sr, sr)(wav)
                        if wav.dim() > 1:
                            wav = wav.mean(dim=0)
                        audio_lengths.append(wav.shape[0])
                        mel = melspec(wav.unsqueeze(0))[0].T
                        mel_lengths.append(mel.shape[0])
                except Exception:
                    continue
        if total > 0:
            print(f"  Analyzed vocoder dataset: {total} files processed")
    if audio_lengths:
        max_audio_len = int(np.ceil(np.percentile(np.array(audio_lengths), audio_percentile) / 256) * 256)
    else:
        max_audio_len = 0
    if mel_lengths:
        max_mel_len = int(np.ceil(np.percentile(np.array(mel_lengths), audio_percentile)))
    else:
        max_mel_len = 0
    return max_audio_len, max_mel_len


class ASRDataset(IterableDataset):
    def __init__(self, csv_path, sr=16000, n_mels=128, cfg=None, shuffle_buffer_size=10000, seed=None, skip_samples=0):
        self.csv_path, self.sr = csv_path, sr
        self.shuffle_buffer_size, self.seed, self.skip_samples = shuffle_buffer_size, seed, skip_samples
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=160, win_length=400, n_mels=n_mels)
        self._num_rows = None
        self.cfg = cfg
        self.warn_on_errors = cfg.get("warn_on_dataset_errors", False) if cfg else False
        self._error_counts = {"missing_file": 0, "load_error": 0, "empty_text": 0, "exceeds_max_len": 0, "ctc_too_short": 0}
        self._first_error_logged = False
        self.max_text_len = cfg.get("max_text_len", None) if cfg else None
        self.max_mel_length = cfg.get("max_mel_length", None) if cfg else None
        self.filter_outliers = cfg.get("filter_outliers", True) if cfg else True
        self.use_augmentation = cfg.get("use_augmentation", True) if cfg else False

    def get_length(self):
        if self._num_rows is None:
            with open(self.csv_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                self._num_rows = sum(1 for _ in reader)
        return self._num_rows

    def get_error_stats(self):
        return self._error_counts.copy()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        rng = random.Random(self.seed + worker_id if self.seed else None)
        buffer = []
        val_split = getattr(self, "_val_split", None)
        val_mode = getattr(self, "_val_mode", False)
        self._error_counts = {"missing_file": 0, "load_error": 0, "empty_text": 0, "exceeds_max_len": 0, "ctc_too_short": 0}
        self._first_error_logged = False
        worker_skip = (self.skip_samples // num_workers) + (1 if worker_id < (self.skip_samples % num_workers) else 0)
        skipped = 0
        with open(self.csv_path, "r", encoding="utf-8", errors="ignore", buffering=8192 * 1024) as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx % num_workers != worker_id:
                    continue
                if val_split and (hash((idx, self.seed)) % 100 < val_split * 100) != val_mode:
                    continue
                if skipped < worker_skip:
                    skipped += 1
                    continue
                wav_path = row.get("wav", "").strip()
                text = row.get("text", "").strip()
                if not wav_path:
                    self._error_counts["load_error"] += 1
                    continue
                if not os.path.exists(wav_path):
                    self._error_counts["missing_file"] += 1
                    continue
                if not text:
                    self._error_counts["empty_text"] += 1
                try:
                    wav, sr = load_audio(wav_path)
                    if sr != self.sr:
                        wav = torchaudio.transforms.Resample(sr, self.sr)(wav)
                    mel = self.melspec(wav)[0].T
                    if self.use_augmentation and not val_mode:
                        T, F = mel.shape
                        if rng.random() < 0.6:
                            speed = rng.uniform(0.9, 1.1)
                            new_T = max(1, int(T * speed))
                            mel = torch.nn.functional.interpolate(
                                mel.unsqueeze(0).unsqueeze(0), size=(new_T, F), mode="bilinear", align_corners=False
                            ).squeeze(0).squeeze(0)
                            T = new_T
                        for _ in range(rng.randint(1, 2)):
                            fw = rng.randint(1, min(8, F // 4))
                            f0 = rng.randint(0, max(1, F - fw))
                            mel[:, f0 : f0 + fw] = 0.0
                        for _ in range(rng.randint(1, 2)):
                            tw = rng.randint(1, min(15, max(1, T // 4)))
                            t0 = rng.randint(0, max(1, T - tw))
                            mel[t0 : t0 + tw, :] = 0.0
                    if self.filter_outliers:
                        skip_sample = False
                        if self.max_text_len is not None and len(text) > self.max_text_len:
                            self._error_counts["exceeds_max_len"] += 1
                            skip_sample = True
                        if self.max_mel_length is not None and mel.shape[0] > self.max_mel_length:
                            self._error_counts["exceeds_max_len"] += 1
                            skip_sample = True
                        downsample_factor = self.cfg.get("downsample_time", 8) if self.cfg else 8
                        output_frames = mel.shape[0] // downsample_factor
                        if len(text) > output_frames:
                            self._error_counts["ctc_too_short"] += 1
                            skip_sample = True
                        if skip_sample:
                            continue
                    if self.shuffle_buffer_size > 0:
                        buffer.append((mel, text))
                        if len(buffer) >= self.shuffle_buffer_size:
                            yield buffer.pop(rng.randint(0, len(buffer) - 1))
                    else:
                        yield mel, text
                except Exception:
                    self._error_counts["load_error"] += 1
                    continue
        if buffer:
            rng.shuffle(buffer)
            yield from buffer
        if self.skip_samples > 0:
            print(f"Dataset exhausted: resetting skip_samples from {self.skip_samples} to 0 for next epoch")
            self.skip_samples = 0


class OCRDataset(IterableDataset):
    def __init__(self, csv_path, image_root, img_size=224, cfg=None, shuffle_buffer_size=10000, seed=None, skip_samples=0, char_to_idx=None, idx_to_char=None):
        self.csv_path, self.image_root, self.img_size = csv_path, image_root, img_size
        self.shuffle_buffer_size, self.seed, self.skip_samples = shuffle_buffer_size, seed, skip_samples
        self.tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
        self._num_rows = None
        if char_to_idx is not None and idx_to_char is not None:
            self.char_to_idx = char_to_idx
            self.idx_to_char = idx_to_char
        else:
            self.char_to_idx, self.idx_to_char = {}, {}
            self._build_vocab(csv_path)
        self.max_text_length = cfg.get("max_text_length", None) if cfg else None
        self.filter_outliers = cfg.get("filter_outliers", True) if cfg else True
        self._error_counts = {"exceeds_max_len": 0}

    def get_error_stats(self):
        return self._error_counts.copy()

    def get_length(self):
        if self._num_rows is None:
            with open(self.csv_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                self._num_rows = sum(1 for _ in reader)
        return self._num_rows

    def _build_vocab(self, csv_path):
        chars = set()
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                chars.update(row.get("text", "") or row.get("label", "") or row.get("text_label", ""))
        self.char_to_idx = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx_to_char = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}
        for char in sorted(chars):
            if char not in self.char_to_idx:
                idx = len(self.char_to_idx)
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        rng = random.Random(self.seed + worker_id if self.seed else None)
        buffer = []
        val_split = getattr(self, "_val_split", None)
        val_mode = getattr(self, "_val_mode", False)
        worker_skip = (self.skip_samples // num_workers) + (1 if worker_id < (self.skip_samples % num_workers) else 0)
        skipped = 0
        with open(self.csv_path, "r", encoding="utf-8", errors="ignore", buffering=8192 * 1024) as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx % num_workers != worker_id:
                    continue
                if val_split and (hash((idx, self.seed)) % 100 < val_split * 100) != val_mode:
                    continue
                if skipped < worker_skip:
                    skipped += 1
                    continue
                img_path = row.get("image") or row.get("img")
                if not img_path:
                    continue
                text = row.get("text", "") or row.get("label", "") or row.get("text_label", "")
                try:
                    full_img_path = os.path.join(self.image_root, img_path) if not os.path.isabs(img_path) else img_path
                    img = Image.open(full_img_path).convert("RGB")
                    text_ids = [self.char_to_idx.get(c, self.char_to_idx["<UNK>"]) for c in text] or [self.char_to_idx["<UNK>"]]
                    final_sequence = [self.char_to_idx["<BOS>"]] + text_ids + [self.char_to_idx["<EOS>"]]
                    if self.filter_outliers and self.max_text_length is not None and len(final_sequence) > self.max_text_length:
                        self._error_counts["exceeds_max_len"] += 1
                        continue
                    result = (self.tf(img), final_sequence)
                    if self.shuffle_buffer_size > 0:
                        buffer.append(result)
                        if len(buffer) >= self.shuffle_buffer_size:
                            yield buffer.pop(rng.randint(0, len(buffer) - 1))
                    else:
                        yield result
                except Exception:
                    continue
        if buffer:
            rng.shuffle(buffer)
            yield from buffer
        if self.skip_samples > 0:
            print(f"Dataset exhausted: resetting skip_samples from {self.skip_samples} to 0 for next epoch")
            self.skip_samples = 0


class TTSDataset(IterableDataset):
    def __init__(self, csv_path, sr=16000, n_mels=128, frame_ms=80, cfg=None, shuffle_buffer_size=10000, seed=None, skip_samples=0):
        self.csv_path, self.sr = csv_path, sr
        self.shuffle_buffer_size, self.seed, self.skip_samples = shuffle_buffer_size, seed, skip_samples
        hop_length = int(sr * frame_ms / 1000)
        win_length = min(1024, hop_length * 4)
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=hop_length, win_length=win_length, n_mels=n_mels)
        self._num_rows = None
        self.max_mel_length = cfg.get("max_mel_length", None) if cfg else None
        self.filter_outliers = cfg.get("filter_outliers", True) if cfg else True
        self._error_counts = {"exceeds_max_len": 0}

    def get_error_stats(self):
        return self._error_counts.copy()

    def get_length(self):
        if self._num_rows is None:
            with open(self.csv_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                self._num_rows = sum(1 for _ in reader)
        return self._num_rows

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        rng = random.Random(self.seed + worker_id if self.seed else None)
        buffer = []
        val_split = getattr(self, "_val_split", None)
        val_mode = getattr(self, "_val_mode", False)
        worker_skip = (self.skip_samples // num_workers) + (1 if worker_id < (self.skip_samples % num_workers) else 0)
        skipped = 0
        with open(self.csv_path, "r", encoding="utf-8", errors="ignore", buffering=8192 * 1024) as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx % num_workers != worker_id:
                    continue
                if val_split and (hash((idx, self.seed)) % 100 < val_split * 100) != val_mode:
                    continue
                if skipped < worker_skip:
                    skipped += 1
                    continue
                try:
                    wav, sr = load_audio(row["wav"])
                    if sr != self.sr:
                        wav = torchaudio.transforms.Resample(sr, self.sr)(wav)
                    mel = self.melspec(wav)[0].T
                    if mel.shape[0] == 0:
                        self._error_counts["zero_mel_len"] = self._error_counts.get("zero_mel_len", 0) + 1
                        continue
                    if self.filter_outliers and self.max_mel_length is not None and mel.shape[0] > self.max_mel_length:
                        self._error_counts["exceeds_max_len"] += 1
                        continue
                    if self.shuffle_buffer_size > 0:
                        buffer.append(mel)
                        if len(buffer) >= self.shuffle_buffer_size:
                            yield buffer.pop(rng.randint(0, len(buffer) - 1))
                    else:
                        yield mel
                except Exception:
                    continue
        if buffer:
            rng.shuffle(buffer)
            yield from buffer
        if self.skip_samples > 0:
            print(f"Dataset exhausted: resetting skip_samples from {self.skip_samples} to 0 for next epoch")
            self.skip_samples = 0


class ImgCapDataset(IterableDataset):
    def __init__(self, manifest, image_root, tokenizer, ctx_len, img_size=224, shuffle_buffer_size=10000, seed=None, skip_samples=0, augment=False):
        self.manifest_path, self.root = manifest, image_root
        self.tok = tokenizer
        self.ctx = ctx_len
        self.shuffle_buffer_size, self.seed, self.skip_samples = shuffle_buffer_size, seed, skip_samples
        self.augment = augment
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.augment:
            self.train_tf = transforms.Compose(
                [
                    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            self.train_tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), normalize])
        self.val_tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), normalize])
        self._num_items = None

    def get_length(self):
        if self._num_items is None:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                items = json.load(f)
                self._num_items = len(items)
        return self._num_items

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        rng = random.Random(self.seed + worker_id if self.seed else None)
        buffer = []
        val_split = getattr(self, "_val_split", None)
        val_mode = getattr(self, "_val_mode", False)
        worker_skip = (self.skip_samples // num_workers) + (1 if worker_id < (self.skip_samples % num_workers) else 0)
        skipped = 0
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        for idx, it in enumerate(items):
            if idx % num_workers != worker_id:
                continue
            if val_split and (hash((idx, self.seed)) % 100 < val_split * 100) != val_mode:
                continue
            if skipped < worker_skip:
                skipped += 1
                continue
            try:
                img = Image.open(os.path.join(self.root, it["image"])).convert("RGB")
                tf = self.val_tf if val_mode else self.train_tf
                image_tensor = tf(img)
                caption = it["caption"].strip()
                ids = [1] + self.tok.encode(caption)
                ids = ids[: self.ctx]
                text_tensor = torch.tensor(ids + [0] * (self.ctx - len(ids)), dtype=torch.long)
                result = (image_tensor, text_tensor)
                if self.shuffle_buffer_size > 0:
                    buffer.append(result)
                    if len(buffer) >= self.shuffle_buffer_size:
                        yield buffer.pop(rng.randint(0, len(buffer) - 1))
                else:
                    yield result
            except Exception:
                continue
        if buffer:
            rng.shuffle(buffer)
            yield from buffer
        if self.skip_samples > 0:
            print(f"Dataset exhausted: resetting skip_samples from {self.skip_samples} to 0 for next epoch")
            self.skip_samples = 0


class VocoderDataset(IterableDataset):
    def __init__(self, csv_path, sr=16000, n_mels=128, n_fft=1024, hop_length=256, cfg=None, shuffle_buffer_size=10000, seed=None, skip_samples=0):
        self.csv_path, self.sr, self.n_mels = csv_path, sr, n_mels
        self.shuffle_buffer_size, self.seed, self.skip_samples = shuffle_buffer_size, seed, skip_samples
        self.max_audio_length = cfg.get("max_audio_length", None) if cfg else None
        self.max_mel_length = cfg.get("max_mel_length", None) if cfg else None
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, n_mels=n_mels)
        self._num_rows = None
        self.filter_outliers = cfg.get("filter_outliers", True) if cfg else True
        self._error_counts = {"exceeds_max_len": 0}

    def get_error_stats(self):
        return self._error_counts.copy()

    def get_length(self):
        if self._num_rows is None:
            with open(self.csv_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                self._num_rows = sum(1 for _ in reader)
        return self._num_rows

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        rng = random.Random(self.seed + worker_id if self.seed else None)
        buffer = []
        val_split = getattr(self, "_val_split", None)
        val_mode = getattr(self, "_val_mode", False)
        worker_skip = (self.skip_samples // num_workers) + (1 if worker_id < (self.skip_samples % num_workers) else 0)
        skipped = 0
        with open(self.csv_path, "r", encoding="utf-8", errors="ignore", buffering=8192 * 1024) as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx % num_workers != worker_id:
                    continue
                if val_split and (hash((idx, self.seed)) % 100 < val_split * 100) != val_mode:
                    continue
                if skipped < worker_skip:
                    skipped += 1
                    continue
                try:
                    audio, sr = load_audio(row["wav"])
                    if sr != self.sr:
                        audio = torchaudio.transforms.Resample(sr, self.sr)(audio)
                    if audio.shape[0] > 1:
                        audio = audio.mean(dim=0, keepdim=True)
                    audio = audio.squeeze(0)
                    if self.filter_outliers and self.max_audio_length is not None and audio.shape[0] > self.max_audio_length:
                        self._error_counts["exceeds_max_len"] += 1
                        continue
                    mel = self.melspec(audio.unsqueeze(0))[0].T
                    if self.filter_outliers and self.max_mel_length is not None and mel.shape[0] > self.max_mel_length:
                        self._error_counts["exceeds_max_len"] += 1
                        continue
                    mel_min, mel_max = mel.min(), mel.max()
                    if mel_max > mel_min + 1e-6:
                        mel = (mel - mel_min) / (mel_max - mel_min + 1e-8)
                    result = (mel, audio)
                    if self.shuffle_buffer_size > 0:
                        buffer.append(result)
                        if len(buffer) >= self.shuffle_buffer_size:
                            yield buffer.pop(rng.randint(0, len(buffer) - 1))
                    else:
                        yield result
                except Exception:
                    continue
        if buffer:
            rng.shuffle(buffer)
            yield from buffer
        if self.skip_samples > 0:
            print(f"Dataset exhausted: resetting skip_samples from {self.skip_samples} to 0 for next epoch")
            self.skip_samples = 0


class MixDataset(IterableDataset):
    def __init__(self, text_path, image_manifest, image_root, asr_csv, ctx=1024, shuffle_buffer_size=10000, seed=None, skip_samples=0):
        self.text_path, self.image_manifest_path, self.image_root, self.asr_csv_path, self.ctx = text_path, image_manifest, image_root, asr_csv, ctx
        self.shuffle_buffer_size, self.seed, self.skip_samples = shuffle_buffer_size, seed, skip_samples
        self.tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self._num_items = None

    def get_length(self):
        if self._num_items is None:
            text_count = sum(1 for _ in open(self.text_path, "r", encoding="utf-8", errors="ignore"))
            with open(self.image_manifest_path, "r", encoding="utf-8") as f:
                image_count = len(json.load(f))
            with open(self.asr_csv_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                asr_count = sum(1 for _ in reader)
            self._num_items = max(text_count, image_count, asr_count)
        return self._num_items

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        rng = random.Random(self.seed + worker_id if self.seed else None)
        buffer = []
        val_split = getattr(self, "_val_split", None)
        val_mode = getattr(self, "_val_mode", False)
        worker_skip = (self.skip_samples // num_workers) + (1 if worker_id < (self.skip_samples % num_workers) else 0)
        skipped = 0
        with open(self.image_manifest_path, "r", encoding="utf-8") as f:
            images = json.load(f)
        with open(self.text_path, "r", encoding="utf-8", errors="ignore", buffering=8192 * 1024) as text_file:
            with open(self.asr_csv_path, "r", encoding="utf-8", errors="ignore", buffering=8192 * 1024) as asr_file:
                asr_reader = csv.DictReader(asr_file)
                for idx, (text_line, img_item, asr_row) in enumerate(zip_longest(text_file, images, asr_reader, fillvalue=None)):
                    if idx % num_workers != worker_id:
                        continue
                    if val_split and (hash((idx, self.seed)) % 100 < val_split * 100) != val_mode:
                        continue
                    if skipped < worker_skip:
                        skipped += 1
                        continue
                    it = {}
                    it["text"] = text_line.strip() if text_line and text_line.strip() else "Describe the image or audio."
                    if img_item:
                        it["image"] = os.path.join(self.image_root, img_item["image"])
                        it["caption"] = img_item.get("caption", "")
                    else:
                        it["image"] = None
                        it["caption"] = ""
                    if asr_row:
                        it["audio"], it["trans"] = asr_row.get("wav", ""), asr_row.get("text", "")
                    else:
                        it["audio"] = None
                        it["trans"] = ""
                    if self.shuffle_buffer_size > 0:
                        buffer.append(it)
                        if len(buffer) >= self.shuffle_buffer_size:
                            yield buffer.pop(rng.randint(0, len(buffer) - 1))
                    else:
                        yield it
        if buffer:
            rng.shuffle(buffer)
            yield from buffer
        if self.skip_samples > 0:
            print(f"Dataset exhausted: resetting skip_samples from {self.skip_samples} to 0 for next epoch")
            self.skip_samples = 0

__all__ = [
    "TextDataset",
    "ASRDataset",
    "OCRDataset",
    "TTSDataset",
    "ImgCapDataset",
    "VocoderDataset",
    "MixDataset",
    "collate_mel_fn",
    "collate_mel_text_fn",
    "collate_mel_audio_fn",
    "build_char_vocab_from_asr_csv",
    "calculate_max_text_len_from_asr_csv",
    "calculate_max_mel_length_from_asr_csv",
    "analyze_asr_dataset",
    "analyze_tts_dataset",
    "analyze_ocr_dataset",
    "analyze_text_dataset",
    "analyze_vocoder_dataset",
]
