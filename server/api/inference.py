"""Inference API — 3 modes: normal (checkpoints), standalone (export), HuggingFace (export).

Supports full multimodal: text, image, video, audio input, audio output, OCR.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

# ── Engine cache (lazy-loaded, one at a time) ────────────────

_engine: Any = None
_engine_key: str | None = None  # "mode:ckpt_dir"
_engine_lock = threading.Lock()


def _unload_engine_unlocked() -> None:
    global _engine, _engine_key
    if _engine is not None:
        try:
            del _engine
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        _engine = None
        _engine_key = None


def _get_normal_engine(ckpt_dir: str, load_talker: bool = False, load_vocoder: bool = False) -> Any:
    """Lazy-load InferenceEngine from checkpoints."""
    global _engine, _engine_key
    key = f"normal:{ckpt_dir}:{load_talker}:{load_vocoder}"

    with _engine_lock:
        if _engine is not None and _engine_key == key:
            return _engine
        _unload_engine_unlocked()

        from test.infer_chat import InferenceEngine
        _engine = InferenceEngine(ckpt_dir=ckpt_dir, device="cuda")
        _engine.load_models(
            load_vision=True,
            load_audio=True,
            load_talker=load_talker,
            load_ocr=True,
            load_vocoder=load_vocoder,
        )
        _engine_key = key
        return _engine


def _get_hf_multimodal(model_dir: str) -> Any:
    """Lazy-load MuOmniMultimodalModel from export dir."""
    global _engine, _engine_key
    key = f"hf_mm:{model_dir}"

    with _engine_lock:
        if _engine is not None and _engine_key == key:
            return _engine
        _unload_engine_unlocked()

        import torch
        from export.modeling_muomni import MuOmniMultimodalModel
        _engine = MuOmniMultimodalModel.from_pretrained_safetensors(model_dir).to("cuda").eval()
        _engine_key = key
        return _engine


def _get_hf_text(model_dir: str) -> Any:
    """Lazy-load MuOmniForCausalLM from export dir."""
    global _engine, _engine_key
    key = f"hf_text:{model_dir}"

    with _engine_lock:
        if _engine is not None and _engine_key == key:
            return _engine
        _unload_engine_unlocked()

        import torch
        from export.modeling_muomni import MuOmniForCausalLM
        _engine = MuOmniForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, torch_dtype=torch.float32
        ).to("cuda").eval()
        _engine_key = key
        return _engine


def unload_engine() -> None:
    """Public unload for VRAM safety — called before training starts."""
    with _engine_lock:
        _unload_engine_unlocked()


def register_unload_callback(pm: Any) -> None:
    pm.register_before_start(unload_engine)


def _check_gpu(handler: Any) -> bool:
    """Return True if GPU is free, else send 409 and return False."""
    from server.app import get_process_manager
    pm = get_process_manager()
    if pm.is_gpu_busy():
        handler.send_error_json(409, "GPU busy with training/testing — cannot run inference")
        return False
    return True


# ── Route handler ────────────────────────────────────────────

def handle_post(handler: Any, path: str, body: dict[str, Any]) -> None:

    # ── Mode 1: Normal inference (from checkpoints) ──────────
    if path == "/api/inference/chat":
        text = body.get("text", "")
        ckpt_dir = body.get("ckpt_dir", "checkpoints/omni_sft_tiny")
        image_path = body.get("image_path")
        video_path = body.get("video_path")
        audio_in = body.get("audio_in")
        audio_out = body.get("audio_out")
        use_ocr = body.get("use_ocr", False)

        if not text and not image_path and not audio_in and not video_path:
            handler.send_error_json(400, "Provide at least one of: text, image_path, audio_in, video_path")
            return

        if not _check_gpu(handler):
            return

        # Determine whether talker/vocoder needed
        need_talker = audio_out is not None
        need_vocoder = audio_out is not None

        try:
            t0 = time.time()
            engine = _get_normal_engine(ckpt_dir, load_talker=need_talker, load_vocoder=need_vocoder)
            response_text = engine.infer(
                text=text or None,
                image=image_path,
                video=video_path,
                audio_in=audio_in,
                use_ocr=use_ocr,
                audio_out=audio_out,
            )
            elapsed_ms = round((time.time() - t0) * 1000)
            handler.send_json({
                "ok": True,
                "mode": "normal",
                "response": response_text,
                "audio_out": audio_out,
                "elapsed_ms": elapsed_ms,
                "ckpt_dir": ckpt_dir,
            })
        except Exception as e:
            handler.send_error_json(500, f"Inference error: {e}")
        return

    # ── Mode 2: Standalone export inference ───────────────────
    if path == "/api/inference/standalone":
        text = body.get("text", "")
        model_dir = body.get("model_dir", "export")
        max_tokens = body.get("max_tokens", 64)

        if not text:
            handler.send_error_json(400, "Missing 'text' field")
            return

        if not Path(model_dir).exists():
            handler.send_error_json(404, f"Export directory not found: {model_dir}")
            return

        if not _check_gpu(handler):
            return

        try:
            t0 = time.time()
            from export.infer_standalone import load_model_with_transformers, simple_generate
            model_data = load_model_with_transformers(model_dir, device="cuda")
            response_text = simple_generate(model_data, text, max_new_tokens=max_tokens, device="cuda")
            elapsed_ms = round((time.time() - t0) * 1000)
            handler.send_json({
                "ok": True,
                "mode": "standalone",
                "response": response_text,
                "elapsed_ms": elapsed_ms,
                "model_dir": model_dir,
            })
        except Exception as e:
            handler.send_error_json(500, f"Standalone inference error: {e}")
        return

    # ── Mode 3: HuggingFace inference ─────────────────────────
    if path == "/api/inference/huggingface":
        text = body.get("text", "")
        model_dir = body.get("model_dir", "export")
        max_tokens = body.get("max_tokens", 32)
        temperature = body.get("temperature", 0.7)
        multimodal = body.get("multimodal", False)
        image_path = body.get("image_path")
        audio_path = body.get("audio_path")

        if not text and not image_path and not audio_path:
            handler.send_error_json(400, "Provide at least one of: text, image_path, audio_path")
            return

        if not Path(model_dir).exists():
            handler.send_error_json(404, f"Export directory not found: {model_dir}")
            return

        if not _check_gpu(handler):
            return

        try:
            import torch
            from omni.tokenizer import BPETokenizer

            t0 = time.time()
            tok_path = os.path.join(model_dir, "tokenizer.model")
            tok = BPETokenizer(tok_path) if os.path.exists(tok_path) else None

            if multimodal or image_path or audio_path:
                # HF multimodal model
                model = _get_hf_multimodal(model_dir)
                results: dict[str, Any] = {"mode": "hf_multimodal"}

                with torch.inference_mode():
                    # Encode image if provided
                    if image_path and os.path.exists(image_path):
                        from torchvision import transforms
                        from PIL import Image
                        img_transforms = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                        ])
                        img = img_transforms(Image.open(image_path).convert("RGB")).unsqueeze(0).to("cuda")
                        img_emb = model.encode_image(img)
                        results["image_encoded"] = True

                        # Generate text from image embedding
                        gen_ids = model.generate_text(img_emb, max_new_tokens=max_tokens, temperature=temperature)
                        if tok:
                            results["image_response"] = tok.decode(gen_ids[0].tolist())

                    # Encode audio if provided
                    if audio_path and os.path.exists(audio_path):
                        import torchaudio
                        waveform, sr = torchaudio.load(audio_path)
                        mel_transform = torchaudio.transforms.MelSpectrogram(
                            sample_rate=sr, n_mels=128, n_fft=1024, hop_length=256
                        )
                        mel = mel_transform(waveform).squeeze(0).T.unsqueeze(0).to("cuda")
                        audio_emb = model.encode_audio(mel)
                        results["audio_encoded"] = True

                        gen_ids = model.generate_text(audio_emb, max_new_tokens=max_tokens, temperature=temperature)
                        if tok:
                            results["audio_response"] = tok.decode(gen_ids[0].tolist())

                    # Text generation if text provided
                    if text and tok:
                        ids = torch.tensor([[1] + tok.encode(text)], device="cuda")
                        output = model(input_ids=ids)
                        next_token = output.logits[:, -1, :].argmax(dim=-1).item()
                        results["next_token"] = tok.decode([next_token])

                results["ok"] = True
                results["elapsed_ms"] = round((time.time() - t0) * 1000)
                handler.send_json(results)

            else:
                # HF text-only model
                model = _get_hf_text(model_dir)

                with torch.inference_mode():
                    if tok:
                        ids = torch.tensor([[1] + tok.encode(text)], device="cuda")
                        gen = model.generate(
                            ids,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_k=40,
                            repetition_penalty=1.3,
                        )
                        response_text = tok.decode(gen[0].tolist())
                    else:
                        response_text = "(tokenizer not found)"

                elapsed_ms = round((time.time() - t0) * 1000)
                handler.send_json({
                    "ok": True,
                    "mode": "hf_text",
                    "response": response_text,
                    "elapsed_ms": elapsed_ms,
                    "model_dir": model_dir,
                })

        except Exception as e:
            handler.send_error_json(500, f"HuggingFace inference error: {e}")
        return

    # ── Unload ────────────────────────────────────────────────
    if path == "/api/inference/unload":
        unload_engine()
        handler.send_json({"ok": True, "unloaded": True})
        return

    handler.send_error_json(404, f"Unknown inference endpoint: {path}")
