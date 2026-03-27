from typing import Sequence

from omni.checkpoint_utils import find_checkpoint


def load_component_checkpoint(ckpt_dir: str, model_file: str, step_prefix: str, device: str = "cpu"):
    """Common helper for export/inference component checkpoint resolution."""
    return find_checkpoint(ckpt_dir, model_file, step_prefix, device)


def extract_prefixed_state(merged_state: dict, prefix: str) -> dict:
    if merged_state is None:
        return {}
    prefix_with_dot = f"{prefix}."
    return {
        k[len(prefix_with_dot):]: v
        for k, v in merged_state.items()
        if k.startswith(prefix_with_dot)
    }


def pick_first_present(state_dict: dict, keys: Sequence[str]):
    for key in keys:
        if key in state_dict:
            return state_dict[key]
    return None
