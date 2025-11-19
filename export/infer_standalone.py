"""
Standalone inference script for μOmni using Hugging Face transformers library.

This script can be used with just the exported model folder and transformers library.
It uses transformers for tokenization and safetensors for model loading.

Usage:
    python infer_standalone.py --model_dir . --text "Hello, how are you?"
    
Or from the export folder:
    python infer_standalone.py --text "Hello, how are you?"
"""

import argparse
import os
import sys
import torch
import json
from pathlib import Path
from safetensors.torch import load_file

try:
    from transformers import AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not found. Install with: pip install transformers")

try:
    from PIL import Image
    import torchvision.transforms as transforms
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

try:
    import torchaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


def load_model_with_transformers(model_dir, device="cuda"):
    """
    Load model using transformers library and safetensors.
    
    Note: This is a simplified version. For full functionality, you may need
    to register custom model classes with transformers or use the full codebase.
    """
    model_dir = Path(model_dir).resolve()
    
    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Load tokenizer using transformers
    if TRANSFORMERS_AVAILABLE:
        try:
            # Try to load tokenizer using transformers
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            print("✓ Loaded tokenizer using transformers")
        except Exception as e:
            print(f"⚠ Could not load tokenizer with transformers: {e}")
            print("  Falling back to SentencePiece directly...")
            try:
                import sentencepiece as spm
                tokenizer_model = model_dir / "tokenizer.model"
                if tokenizer_model.exists():
                    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
                    tokenizer = SimpleTokenizer(sp)
                    print("✓ Loaded tokenizer using SentencePiece")
                else:
                    raise FileNotFoundError(f"tokenizer.model not found in {model_dir}")
            except ImportError:
                raise ImportError("sentencepiece not installed. Install with: pip install sentencepiece")
    else:
        # Fallback to SentencePiece
        import sentencepiece as spm
        tokenizer_model = model_dir / "tokenizer.model"
        if not tokenizer_model.exists():
            raise FileNotFoundError(f"tokenizer.model not found in {model_dir}")
        sp = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
        tokenizer = SimpleTokenizer(sp)
        print("✓ Loaded tokenizer using SentencePiece")
    
    # Load model weights from safetensors
    safetensors_path = model_dir / "model.safetensors"
    if not safetensors_path.exists():
        # Try alternative names
        for alt_name in ["muomni.safetensors", "omni.safetensors"]:
            alt_path = model_dir / alt_name
            if alt_path.exists():
                safetensors_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Could not find safetensors file in {model_dir}")
    
    print(f"Loading model weights from {safetensors_path}...")
    state_dict = load_file(str(safetensors_path), device=device)
    print(f"  Loaded {len(state_dict)} parameter tensors")
    
    # Extract component info
    components = {}
    prefixes = ["thinker", "vision_encoder", "audio_encoder", "talker", "rvq", "proj_a", "proj_v", "ocr"]
    for prefix in prefixes:
        component_keys = [k for k in state_dict.keys() if k.startswith(prefix + ".")]
        if component_keys:
            components[prefix] = len(component_keys)
            print(f"  Found {prefix}: {len(component_keys)} parameters")
    
    return {
        "config": config,
        "state_dict": state_dict,
        "tokenizer": tokenizer,
        "model_dir": model_dir,
        "components": components
    }


class SimpleTokenizer:
    """Simple wrapper for SentencePiece tokenizer"""
    def __init__(self, sp_processor):
        self.sp = sp_processor
    
    def encode(self, text):
        """Encode text to token IDs"""
        return self.sp.encode(text, out_type=int)
    
    def decode(self, token_ids):
        """Decode token IDs to text"""
        if isinstance(token_ids, list):
            return self.sp.decode(token_ids)
        return self.sp.decode(token_ids.tolist() if hasattr(token_ids, 'tolist') else [token_ids])


def simple_generate(model_data, prompt, max_new_tokens=64, device="cuda"):
    """
    Simple text generation using the loaded model.
    
    Note: This is a simplified version. For full multimodal inference (image, audio, video),
    you need to use the full codebase with infer_chat.py or register custom model classes
    with transformers.
    
    This function demonstrates the structure but requires the model architecture classes
    from the omni package to actually generate text.
    """
    tokenizer = model_data["tokenizer"]
    state_dict = model_data["state_dict"]
    config = model_data["config"]
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = [1] + input_ids  # Add BOS token (token ID 1)
    
    print(f"Input tokens: {len(input_ids)}")
    print(f"Prompt: {prompt}")
    print(f"Model config: vocab_size={config.get('vocab_size', 'N/A')}, "
          f"d_model={config.get('d_model', 'N/A')}, "
          f"n_layers={config.get('n_layers', 'N/A')}")
    
    # For actual generation, you would need to:
    # 1. Import model architecture classes: from omni.thinker import ThinkerLM
    # 2. Initialize model with config
    # 3. Load weights from state_dict (with proper prefix handling)
    # 4. Run forward passes with KV caching
    
    # This is a placeholder that shows the structure
    print("\n⚠ Note: Full generation requires model architecture classes from the codebase.")
    print("   For full functionality, use:")
    print("   1. infer_chat.py (from root directory) - Full multimodal inference")
    print("   2. Register custom model classes with transformers")
    print("   3. Use Hugging Face's model loading with custom architecture definitions")
    
    # Return a simple response
    return f"[Generation requires model architecture. To generate text, use infer_chat.py from the root directory. Input: {prompt}]"


def main():
    parser = argparse.ArgumentParser(
        description="Standalone inference for μOmni using transformers library"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=".",
        help="Directory containing model.safetensors and config files (default: current directory)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate (default: 64)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("μOmni Standalone Inference (using transformers)")
    print("=" * 60)
    print(f"Model directory: {args.model_dir}")
    print(f"Device: {args.device}")
    print()
    
    # Load model
    try:
        model_data = load_model_with_transformers(args.model_dir, device=args.device)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        print("\nMake sure:")
        print("  1. You're in the export folder or specify --model_dir")
        print("  2. All required files are present (model.safetensors, config.json, tokenizer.model)")
        print("  3. Required libraries are installed: transformers, safetensors, sentencepiece")
        sys.exit(1)
    
    # Generate
    if args.text:
        prompt = args.text
    else:
        prompt = input("Enter your prompt: ")
    
    print("\n" + "=" * 60)
    print("Generation")
    print("=" * 60)
    
    output = simple_generate(model_data, prompt, max_new_tokens=args.max_tokens, device=args.device)
    print(f"\nOutput: {output}")
    
    print("\n" + "=" * 60)
    print("Note: This is a simplified standalone version using transformers.")
    print("For full multimodal inference (image, audio, video), you may need")
    print("to register custom model classes with transformers or use the full codebase.")
    print("=" * 60)


if __name__ == "__main__":
    main()

