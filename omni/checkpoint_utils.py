import json
import os

import torch


def find_checkpoint(checkpoint_dir, standard_name, step_prefix, device="gpu"):
    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        return None, None

    standard_path = os.path.join(checkpoint_dir, standard_name)
    if os.path.exists(standard_path):
        try:
            checkpoint = torch.load(standard_path, map_location=device)
            print(f"Using standard checkpoint: {standard_name}")
            return standard_path, checkpoint
        except Exception as e:
            print(f"Warning: Could not load {standard_path}: {e}")

    if standard_name != "model.pt":
        model_pt_path = os.path.join(checkpoint_dir, "model.pt")
        if os.path.exists(model_pt_path):
            try:
                checkpoint = torch.load(model_pt_path, map_location=device)
                print("Using standard checkpoint: model.pt")
                return model_pt_path, checkpoint
            except Exception as e:
                print(f"Warning: Could not load {model_pt_path}: {e}")

    if step_prefix:
        try:
            checkpoint_files = [
                f for f in os.listdir(checkpoint_dir)
                if f.startswith(step_prefix) and f.endswith(".pt")
            ]
            if checkpoint_files:
                step_numbers = []
                for f in checkpoint_files:
                    try:
                        step_num = int(f.replace(step_prefix, "").replace(".pt", ""))
                        step_numbers.append((step_num, f))
                    except Exception:
                        continue
                if step_numbers:
                    step_numbers.sort(key=lambda x: x[0], reverse=True)
                    latest_path = os.path.join(checkpoint_dir, step_numbers[0][1])
                    checkpoint = torch.load(latest_path, map_location=device)
                    print(f"Using step checkpoint: {step_numbers[0][1]} (step {step_numbers[0][0]})")
                    return latest_path, checkpoint
        except Exception as e:
            print(f"Warning: Could not search for step checkpoints: {e}")
    return None, None


def strip_orig_mod(state_dict):
    if state_dict is None:
        return None
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("._orig_mod.", ".").replace("_orig_mod.", "")
        new_state_dict[new_key] = value
    return new_state_dict


def convert_attention_weights(state_dict):
    if state_dict is None:
        return None
    new_state_dict = {}
    processed_keys = set()
    for key, value in state_dict.items():
        if key.endswith('.q.weight') or key.endswith('.k.weight') or key.endswith('.v.weight'):
            base_key = key.rsplit('.', 2)[0]
            q_key = f"{base_key}.q.weight"
            k_key = f"{base_key}.k.weight"
            v_key = f"{base_key}.v.weight"
            if q_key in state_dict and k_key in state_dict and v_key in state_dict:
                if base_key not in processed_keys:
                    qkv_weight = torch.cat([state_dict[q_key], state_dict[k_key], state_dict[v_key]], dim=0)
                    new_state_dict[f"{base_key}.qkv.weight"] = qkv_weight
                    processed_keys.update({base_key, q_key, k_key, v_key})
                    continue
        if key in processed_keys:
            continue
        new_state_dict[key] = value
    return new_state_dict


def normalize_state_dict(state_dict, strip_orig_mod_prefix=True, convert_attention=True):
    if state_dict is None:
        return None
    result = state_dict
    if strip_orig_mod_prefix:
        result = strip_orig_mod(result)
    if convert_attention:
        result = convert_attention_weights(result)
    return result


def save_training_metadata(save_dir, model_name, metadata):
    os.makedirs(save_dir, exist_ok=True)
    metadata_path = os.path.join(save_dir, f"{model_name}_metadata.json")
    serializable_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, dict):
            serializable_metadata[key] = {str(k): v for k, v in value.items()}
        elif isinstance(value, (int, float, str, bool, list, type(None))):
            serializable_metadata[key] = value
        elif isinstance(value, torch.Tensor):
            serializable_metadata[key] = value.tolist() if value.numel() < 100 else None
        else:
            serializable_metadata[key] = str(value)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(serializable_metadata, f, indent=2, ensure_ascii=False)
    return metadata_path


def load_training_metadata(save_dir, model_name):
    metadata_path = os.path.join(save_dir, f"{model_name}_metadata.json")
    if not os.path.exists(metadata_path):
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load metadata from {metadata_path}: {e}")
        return None


def load_checkpoint(save_dir, model_name, device, logger, state_dict_loaders=None):
    if not os.path.exists(save_dir):
        return 0, None
    metadata = load_training_metadata(save_dir, model_name)
    if metadata is None:
        return 0, None
    step = metadata.get("step", 0)
    if step == 0:
        return 0, None
    logger.info(f"Found checkpoint at step {step}, loading from metadata")
    model_path = os.path.join(save_dir, f"{model_name}.pt")
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and state_dict_loaders:
                for key, (_, load_func) in state_dict_loaders.items():
                    if key in checkpoint:
                        try:
                            load_func(checkpoint[key])
                        except Exception as e:
                            logger.warning(f"Failed to load {key}: {e}")
            logger.info(f"Loaded model weights from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            return 0, None
    else:
        logger.warning(f"Model file not found: {model_path}, but metadata exists")
        return step, metadata
    return step, metadata

__all__ = [
    "find_checkpoint",
    "strip_orig_mod",
    "normalize_state_dict",
    "save_training_metadata",
    "load_training_metadata",
    "load_checkpoint",
]
