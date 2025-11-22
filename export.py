"""
Script to merge all μOmni model components into a single safetensors file.

This script collects weights from:
- Thinker (LLM)
- Audio Encoder
- Vision Encoder  
- Talker (TTS) + RVQ
- Multimodal Projectors (proj_a, proj_v)
- Optional: OCR model

All weights are merged into a single safetensors file with prefixed keys
to avoid naming conflicts.
"""

import argparse
import json
import os
import shutil
import torch
from safetensors.torch import save_file
from pathlib import Path
from omni.utils import find_checkpoint, strip_orig_mod
from typing import Dict


def load_checkpoint(path, device="cpu"):
    """Load a PyTorch checkpoint file."""
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location=device)
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return None


def load_checkpoint_paths_from_config(config_path, configs_dir="configs"):
    """
    Load checkpoint paths from the omni config file.
    
    Args:
        config_path: Path to the omni config file (e.g., "omni_sft_tiny.json")
        configs_dir: Directory containing config files (default: "configs")
    
    Returns:
        dict with keys: omni_ckpt_dir, thinker_ckpt_dir, audio_ckpt_dir, 
                       vision_ckpt_dir, talker_ckpt_dir, ocr_ckpt_dir
    """
    full_config_path = os.path.join(configs_dir, config_path)
    
    if not os.path.exists(full_config_path):
        print(f"Warning: Config file not found: {full_config_path}")
        return {}
    
    try:
        with open(full_config_path, 'r') as f:
            config = json.load(f)
        
        paths = {}
        
        # Get omni checkpoint from save_dir
        if "save_dir" in config:
            paths["omni_ckpt_dir"] = config["save_dir"]
        
        # Get component checkpoints
        if "thinker_ckpt" in config:
            paths["thinker_ckpt_dir"] = config["thinker_ckpt"]
        if "audio_ckpt" in config:
            paths["audio_ckpt_dir"] = config["audio_ckpt"]
        if "vision_ckpt" in config:
            paths["vision_ckpt_dir"] = config["vision_ckpt"]
        if "talker_ckpt" in config:
            paths["talker_ckpt_dir"] = config["talker_ckpt"]
        if "ocr_ckpt" in config:
            paths["ocr_ckpt_dir"] = config["ocr_ckpt"]
        
        print(f"Loaded checkpoint paths from {full_config_path}")
        return paths
    
    except Exception as e:
        print(f"Warning: Could not load config file {full_config_path}: {e}")
        return {}


def merge_model_components(
    omni_ckpt_dir,
    thinker_ckpt_dir=None,
    audio_ckpt_dir=None,
    vision_ckpt_dir=None,
    talker_ckpt_dir=None,
    ocr_ckpt_dir=None,
    output_path="model.safetensors",
    include_optimizer=False
):
    """
    Merge all model components into a single safetensors file.
    
    Args:
        omni_ckpt_dir: Directory containing omni.pt (with projectors)
        thinker_ckpt_dir: Directory containing thinker.pt
        audio_ckpt_dir: Directory containing audio_enc.pt
        vision_ckpt_dir: Directory containing vision.pt
        talker_ckpt_dir: Directory containing talker.pt
        ocr_ckpt_dir: Optional directory containing ocr.pt
        output_path: Path to save the merged safetensors file
        include_optimizer: Whether to include optimizer/scheduler states
    """
    merged_state = {}
    
    print("Merging model components...")
    
    # 1. Load Thinker (LLM)
    # Try from omni checkpoint first, then thinker checkpoint
    thinker_loaded = False
    if omni_ckpt_dir:
        omni_path, omni_ckpt = find_checkpoint(omni_ckpt_dir, "omni.pt", "omni_step_", device="cpu")
        if omni_ckpt and isinstance(omni_ckpt, dict) and "thinker" in omni_ckpt:
            thinker_state = strip_orig_mod(omni_ckpt["thinker"])
            for key, value in thinker_state.items():
                merged_state[f"thinker.{key}"] = value
            thinker_loaded = True
            print(f"  ✓ Loaded Thinker from {omni_path}")
    
    if not thinker_loaded and thinker_ckpt_dir:
        thinker_path, thinker_ckpt = find_checkpoint(thinker_ckpt_dir, "thinker.pt", "thinker_step_", device="cpu")
        if thinker_ckpt:
            # Handle both dict format and direct state_dict
            if isinstance(thinker_ckpt, dict) and "model" in thinker_ckpt:
                thinker_state = strip_orig_mod(thinker_ckpt["model"])
            elif isinstance(thinker_ckpt, dict) and "thinker" in thinker_ckpt:
                thinker_state = strip_orig_mod(thinker_ckpt["thinker"])
            elif isinstance(thinker_ckpt, dict):
                thinker_state = strip_orig_mod(thinker_ckpt)
            else:
                thinker_state = strip_orig_mod(thinker_ckpt)
            
            for key, value in thinker_state.items():
                merged_state[f"thinker.{key}"] = value
            thinker_loaded = True
            print(f"  ✓ Loaded Thinker from {thinker_path}")
    
    if not thinker_loaded:
        print("  ⚠ Warning: Thinker checkpoint not found")
    
    # 2. Load Audio Encoder
    if audio_ckpt_dir:
        audio_path, audio_ckpt = find_checkpoint(audio_ckpt_dir, "audio_enc.pt", "audio_enc_step_", device="cpu")
        if audio_ckpt:
            if isinstance(audio_ckpt, dict) and "enc" in audio_ckpt:
                audio_state = strip_orig_mod(audio_ckpt["enc"])
            elif isinstance(audio_ckpt, dict) and "model" in audio_ckpt:
                audio_state = strip_orig_mod(audio_ckpt["model"])
            else:
                audio_state = strip_orig_mod(audio_ckpt)
            
            for key, value in audio_state.items():
                merged_state[f"audio_encoder.{key}"] = value
            print(f"  ✓ Loaded Audio Encoder from {audio_path}")
        else:
            print(f"  ⚠ Warning: Audio encoder checkpoint not found")
    else:
        print("  ⚠ Warning: Audio encoder checkpoint directory not provided")
    
    # 3. Load Vision Encoder
    if vision_ckpt_dir:
        vision_path, vision_ckpt = find_checkpoint(vision_ckpt_dir, "vision.pt", "vision_step_", device="cpu")
        if vision_ckpt:
            if isinstance(vision_ckpt, dict) and "vit" in vision_ckpt:
                vision_state = strip_orig_mod(vision_ckpt["vit"])
            elif isinstance(vision_ckpt, dict) and "model" in vision_ckpt:
                vision_state = strip_orig_mod(vision_ckpt["model"])
            else:
                vision_state = strip_orig_mod(vision_ckpt)
            
            for key, value in vision_state.items():
                merged_state[f"vision_encoder.{key}"] = value
            print(f"  ✓ Loaded Vision Encoder from {vision_path}")
        else:
            print(f"  ⚠ Warning: Vision encoder checkpoint not found")
    else:
        print("  ⚠ Warning: Vision encoder checkpoint directory not provided")
    
    # 4. Load Talker + RVQ
    if talker_ckpt_dir:
        talker_path, talker_ckpt = find_checkpoint(talker_ckpt_dir, "talker.pt", "talker_step_", device="cpu")
        if talker_ckpt:
            # Load RVQ
            if isinstance(talker_ckpt, dict) and "rvq" in talker_ckpt:
                rvq_state = strip_orig_mod(talker_ckpt["rvq"])
                for key, value in rvq_state.items():
                    merged_state[f"rvq.{key}"] = value
                print(f"  ✓ Loaded RVQ from {talker_path}")
            
            # Load Talker
            if isinstance(talker_ckpt, dict) and "talker" in talker_ckpt:
                talker_state = strip_orig_mod(talker_ckpt["talker"])
                for key, value in talker_state.items():
                    merged_state[f"talker.{key}"] = value
                print(f"  ✓ Loaded Talker from {talker_path}")
        else:
            print(f"  ⚠ Warning: Talker checkpoint not found")
    else:
        print("  ⚠ Warning: Talker checkpoint directory not provided")
    
    # 5. Load Multimodal Projectors (from omni checkpoint)
    if omni_ckpt_dir:
        omni_path, omni_ckpt = find_checkpoint(omni_ckpt_dir, "omni.pt", "omni_step_", device="cpu")
        if omni_ckpt and isinstance(omni_ckpt, dict):
            if "proj_a" in omni_ckpt:
                proj_a_state = strip_orig_mod(omni_ckpt["proj_a"])
                for key, value in proj_a_state.items():
                    merged_state[f"proj_a.{key}"] = value
                print(f"  ✓ Loaded Audio Projector from {omni_path}")
            
            if "proj_v" in omni_ckpt:
                proj_v_state = strip_orig_mod(omni_ckpt["proj_v"])
                for key, value in proj_v_state.items():
                    merged_state[f"proj_v.{key}"] = value
                print(f"  ✓ Loaded Vision Projector from {omni_path}")
        else:
            print(f"  ⚠ Warning: Omni checkpoint not found")
    
    # 6. Load OCR (optional)
    if ocr_ckpt_dir:
        ocr_path, ocr_ckpt = find_checkpoint(ocr_ckpt_dir, "ocr.pt", "ocr_step_", device="cpu")
        if ocr_ckpt:
            if isinstance(ocr_ckpt, dict) and "model" in ocr_ckpt:
                ocr_state = strip_orig_mod(ocr_ckpt["model"])
            elif isinstance(ocr_ckpt, dict):
                ocr_state = strip_orig_mod(ocr_ckpt)
            else:
                ocr_state = strip_orig_mod(ocr_ckpt)
            
            for key, value in ocr_state.items():
                merged_state[f"ocr.{key}"] = value
            print(f"  ✓ Loaded OCR model from {ocr_path}")
        else:
            print(f"  ⚠ Warning: OCR checkpoint not found")
    
    # Save to safetensors
    if not merged_state:
        raise ValueError("No model weights found to merge!")
    
    print(f"\nSaving merged model to {output_path}...")
    print(f"  Total parameters: {len(merged_state)}")
    
    # Convert all tensors to CPU and ensure they're contiguous
    for key in merged_state:
        if isinstance(merged_state[key], torch.Tensor):
            merged_state[key] = merged_state[key].cpu().contiguous()
    
    save_file(merged_state, output_path)
    print(f"  ✓ Saved {output_path}")
    
    # Calculate file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    
    return merged_state


def copy_support_files(
    omni_ckpt_dir,
    thinker_ckpt_dir,
    audio_ckpt_dir,
    vision_ckpt_dir,
    talker_ckpt_dir,
    ocr_ckpt_dir,
    output_dir,
    configs_dir="configs"
):
    """
    Copy all support files needed for inference.
    
    Support files:
    - Config JSON files
    - Tokenizer model
    - Optional: HiFi-GAN vocoder checkpoint
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCopying support files to {output_dir}...")
    
    # Note: Component config files are NOT copied anymore
    # Only the main config.json at root is created (see below)
    # This keeps the export directory clean and Hugging Face compatible
    
    # Copy tokenizer
    tokenizer_paths = [
        os.path.join(thinker_ckpt_dir, "tokenizer.model") if thinker_ckpt_dir else None,
        os.path.join(omni_ckpt_dir, "tokenizer.model") if omni_ckpt_dir else None,
    ]
    
    tokenizer_copied = False
    for tok_path in tokenizer_paths:
        if tok_path and os.path.exists(tok_path):
            shutil.copy2(tok_path, os.path.join(output_dir, "tokenizer.model"))
            print(f"  ✓ Copied tokenizer.model from {tok_path}")
            tokenizer_copied = True
            break
    
    if not tokenizer_copied:
        print("  ⚠ Warning: tokenizer.model not found")
    
    # Copy HiFi-GAN vocoder if available
    hifigan_paths = [
        os.path.join(talker_ckpt_dir, "hifigan.pt") if talker_ckpt_dir else None,
        os.path.join(omni_ckpt_dir, "hifigan.pt") if omni_ckpt_dir else None,
        "checkpoints/hifigan.pt",
    ]
    
    for hifigan_path in hifigan_paths:
        if hifigan_path and os.path.exists(hifigan_path):
            shutil.copy2(hifigan_path, os.path.join(output_dir, "hifigan.pt"))
            print(f"  ✓ Copied hifigan.pt from {hifigan_path}")
            break
    
    # Copy OCR checkpoint if available (contains char mappings)
    # Also try to find step checkpoints
    if ocr_ckpt_dir:
        ocr_path = os.path.join(ocr_ckpt_dir, "ocr.pt")
        if not os.path.exists(ocr_path):
            # Try to find step checkpoint
            _, ocr_ckpt = find_checkpoint(ocr_ckpt_dir, "ocr.pt", "ocr_step_", device="cpu")
            if ocr_ckpt:
                # Save the checkpoint with char mappings
                ocr_path = os.path.join(output_dir, "ocr.pt")
                torch.save(ocr_ckpt, ocr_path)
                print(f"  ✓ Saved ocr.pt (with char mappings) from step checkpoint")
        else:
            shutil.copy2(ocr_path, os.path.join(output_dir, "ocr.pt"))
            print(f"  ✓ Copied ocr.pt (with char mappings) from {ocr_path}")
    
    # Create Hugging Face compatible config.json at root
    # Load thinker config to get model parameters
    thinker_cfg_path = os.path.join(configs_dir, "thinker_tiny.json")
    if not os.path.exists(thinker_cfg_path):
        # Try default location
        default_cfg_path = "configs/thinker_tiny.json"
        if os.path.exists(default_cfg_path):
            thinker_cfg_path = default_cfg_path
        else:
            # Last resort: try in output_dir/configs
            thinker_cfg_path = os.path.join(output_dir, "configs", "thinker_tiny.json")
    
    thinker_cfg = {}
    if os.path.exists(thinker_cfg_path):
        with open(thinker_cfg_path, "r") as f:
            thinker_cfg = json.load(f)
    
    # Create main config.json (compatible with both Hugging Face and our inference script)
    # Load other component configs if available (from configs_dir, not from export directory)
    audio_cfg = {}
    vision_cfg = {}
    talker_cfg = {}
    ocr_cfg = {}
    
    # Load component configs from source configs_dir (not copying them, just reading)
    configs_src = Path(configs_dir)
    if configs_src.exists():
        for cfg_file, cfg_dict in [
            ("audio_enc_tiny.json", audio_cfg),
            ("vision_tiny.json", vision_cfg),
            ("talker_tiny.json", talker_cfg),
            ("ocr_tiny.json", ocr_cfg)
        ]:
            cfg_path = configs_src / cfg_file
            if cfg_path.exists():
                with open(cfg_path, "r") as f:
                    cfg_dict.update(json.load(f))
    
    main_config = {
        "model_type": "muomni",
        "architectures": ["ThinkerLM"],
        # Thinker config (main model)
        "thinker": {
            "vocab_size": thinker_cfg.get("vocab_size", 32000),
            "n_layers": thinker_cfg.get("n_layers", 4),
            "d_model": thinker_cfg.get("d_model", 256),
            "n_heads": thinker_cfg.get("n_heads", 4),
            "d_ff": thinker_cfg.get("d_ff", 1024),
            "dropout": thinker_cfg.get("dropout", 0.1),
            "rope_theta": thinker_cfg.get("rope_theta", 10000),
            "ctx_len": thinker_cfg.get("ctx_len", 512),
            "use_gqa": thinker_cfg.get("use_gqa", False),
            "use_swiglu": thinker_cfg.get("use_swiglu", True),
            "use_moe": thinker_cfg.get("use_moe", False),
            "num_experts": thinker_cfg.get("num_experts", 8),
            "num_experts_per_tok": thinker_cfg.get("num_experts_per_tok", 2)
        },
        # Also include flat structure for Hugging Face compatibility
        "vocab_size": thinker_cfg.get("vocab_size", 32000),
        "n_layers": thinker_cfg.get("n_layers", 4),
        "d_model": thinker_cfg.get("d_model", 256),
        "n_heads": thinker_cfg.get("n_heads", 4),
        "d_ff": thinker_cfg.get("d_ff", 1024),
        "dropout": thinker_cfg.get("dropout", 0.1),
        "rope_theta": thinker_cfg.get("rope_theta", 10000),
        "ctx_len": thinker_cfg.get("ctx_len", 512),
        "use_gqa": thinker_cfg.get("use_gqa", False),
        "use_swiglu": thinker_cfg.get("use_swiglu", True),
        "use_moe": thinker_cfg.get("use_moe", False),
        "num_experts": thinker_cfg.get("num_experts", 8),
        "num_experts_per_tok": thinker_cfg.get("num_experts_per_tok", 2),
        # Component configs (for our inference script)
        "audio": audio_cfg if audio_cfg else None,
        "vision": vision_cfg if vision_cfg else None,
        "talker": talker_cfg if talker_cfg else None,
        "ocr": ocr_cfg if ocr_cfg else None,
        # Hugging Face metadata
        "torch_dtype": "float16",
        "transformers_version": "4.40.0"
    }
    
    # Remove None values
    main_config = {k: v for k, v in main_config.items() if v is not None}
    
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(main_config, f, indent=2)
    print(f"  ✓ Created config.json (Hugging Face format)")
    
    # Create tokenizer_config.json (for SentencePiece)
    tokenizer_config = {
        "tokenizer_class": "SentencePieceTokenizer",
        "model_type": "muomni",
        "vocab_size": thinker_cfg.get("vocab_size", 32000),
        "model_file": "tokenizer.model",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "unk_token": "<UNK>",
        "pad_token": "<PAD>",
        "clean_up_spaces": True
    }
    
    tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(tokenizer_config_path, "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"  ✓ Created tokenizer_config.json")
    
    # Create generation_config.json
    generation_config = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "max_new_tokens": 512,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0
    }
    
    gen_config_path = os.path.join(output_dir, "generation_config.json")
    with open(gen_config_path, "w") as f:
        json.dump(generation_config, f, indent=2)
    print(f"  ✓ Created generation_config.json")
    
    # Create chat_template.json (basic template)
    chat_template = {
        "template": "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}{{ 'Assistant: ' }}"
    }
    
    chat_template_path = os.path.join(output_dir, "chat_template.json")
    with open(chat_template_path, "w") as f:
        json.dump(chat_template, f, indent=2)
    print(f"  ✓ Created chat_template.json")
    
    # Create preprocessor_config.json (for multimodal models)
    preprocessor_config = {
        "image_processor": {
            "size": {"height": 224, "width": 224},
            "resample": "bilinear"
        },
        "audio_processor": {
            "sample_rate": 16000,
            "n_fft": 1024,
            "hop_length": 160,
            "n_mels": 128
        }
    }
    
    preprocessor_config_path = os.path.join(output_dir, "preprocessor_config.json")
    with open(preprocessor_config_path, "w") as f:
        json.dump(preprocessor_config, f, indent=2)
    print(f"  ✓ Created preprocessor_config.json")
    
    # Create model_info.json (custom metadata)
    model_info = {
        "model_type": "muomni",
        "components": {
            "thinker": thinker_ckpt_dir is not None,
            "audio_encoder": audio_ckpt_dir is not None,
            "vision_encoder": vision_ckpt_dir is not None,
            "talker": talker_ckpt_dir is not None,
            "ocr": ocr_ckpt_dir is not None,
        },
        "configs_dir": "configs",
        "tokenizer": "tokenizer.model",
        "safetensors_file": os.path.basename(output_dir) + ".safetensors" if output_dir else "model.safetensors"
    }
    
    info_path = os.path.join(output_dir, "model_info.json")
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"  ✓ Created model_info.json")
    
    print(f"\n✓ All support files copied to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge μOmni model components into a single safetensors file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all components (auto-detects checkpoint paths from configs/omni_sft_tiny.json)
  python export.py

  # Export with custom config file
  python export.py --omni_config omni_sft_tiny.json

  # Export with explicit checkpoint paths (overrides config file)
  python export.py \\
      --omni_ckpt checkpoints/omni_sft_tiny \\
      --thinker_ckpt checkpoints/thinker_tiny \\
      --audio_ckpt checkpoints/audio_enc_tiny \\
      --vision_ckpt checkpoints/vision_tiny \\
      --talker_ckpt checkpoints/talker_tiny

  # Export (only main config.json is created, component configs are not copied)

Note: Config files are always read from the configs/ folder.
      Merged model and export files are saved to the export/ folder by default.
        """
    )
    parser.add_argument(
        "--omni_ckpt",
        type=str,
        help="Path to omni checkpoint directory (contains omni.pt with projectors)"
    )
    parser.add_argument(
        "--thinker_ckpt",
        type=str,
        help="Path to thinker checkpoint directory (contains thinker.pt)"
    )
    parser.add_argument(
        "--audio_ckpt",
        type=str,
        help="Path to audio encoder checkpoint directory (contains audio_enc.pt)"
    )
    parser.add_argument(
        "--vision_ckpt",
        type=str,
        help="Path to vision encoder checkpoint directory (contains vision.pt)"
    )
    parser.add_argument(
        "--talker_ckpt",
        type=str,
        help="Path to talker checkpoint directory (contains talker.pt)"
    )
    parser.add_argument(
        "--ocr_ckpt",
        type=str,
        default=None,
        help="Optional: Path to OCR checkpoint directory (contains ocr.pt)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="export",
        help="Output directory for merged model and support files (default: export)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="model.safetensors",
        help="Output safetensors filename (will be placed in output_dir)"
    )
    parser.add_argument(
        "--configs_dir",
        type=str,
        default="configs",
        help="Directory containing config JSON files (default: configs, always reads from here)"
    )
    parser.add_argument(
        "--omni_config",
        type=str,
        default="omni_sft_tiny.json",
        help="Name of the omni config file in configs_dir to read checkpoint paths from (default: omni_sft_tiny.json)"
    )
    
    args = parser.parse_args()
    
    # Always use configs_dir for reading config files
    configs_dir = args.configs_dir
    
    # Load checkpoint paths from config file if not provided via command line
    config_paths = {}
    if args.omni_config:
        config_paths = load_checkpoint_paths_from_config(args.omni_config, configs_dir)
    
    # Use command line arguments if provided, otherwise use paths from config file
    omni_ckpt_dir = args.omni_ckpt or config_paths.get("omni_ckpt_dir")
    thinker_ckpt_dir = args.thinker_ckpt or config_paths.get("thinker_ckpt_dir")
    audio_ckpt_dir = args.audio_ckpt or config_paths.get("audio_ckpt_dir")
    vision_ckpt_dir = args.vision_ckpt or config_paths.get("vision_ckpt_dir")
    talker_ckpt_dir = args.talker_ckpt or config_paths.get("talker_ckpt_dir")
    ocr_ckpt_dir = args.ocr_ckpt or config_paths.get("ocr_ckpt_dir")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)
    
    # Merge model components
    merged_state = merge_model_components(
        omni_ckpt_dir=omni_ckpt_dir,
        thinker_ckpt_dir=thinker_ckpt_dir,
        audio_ckpt_dir=audio_ckpt_dir,
        vision_ckpt_dir=vision_ckpt_dir,
        talker_ckpt_dir=talker_ckpt_dir,
        ocr_ckpt_dir=ocr_ckpt_dir,
        output_path=output_path
    )
    
    # Copy support files (always use configs_dir for reading configs)
    copy_support_files(
        omni_ckpt_dir=omni_ckpt_dir,
        thinker_ckpt_dir=thinker_ckpt_dir,
        audio_ckpt_dir=audio_ckpt_dir,
        vision_ckpt_dir=vision_ckpt_dir,
        talker_ckpt_dir=talker_ckpt_dir,
        ocr_ckpt_dir=ocr_ckpt_dir,
        output_dir=args.output_dir,
        configs_dir=configs_dir
    )
    
    print(f"\n✓ Model merge complete!")
    print(f"  Model: {output_path}")
    print(f"  Support files: {args.output_dir}/")
    print(f"\nTo use this model, point your inference script to:")
    print(f"  --model_dir {args.output_dir}")


if __name__ == "__main__":
    main()

