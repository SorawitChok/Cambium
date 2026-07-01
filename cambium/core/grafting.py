"""
Lightweight helpers for grafting a single block from a remote model.

Only the source model config and the shards containing the requested
block are downloaded; the rest of the source weights stay on the hub.
"""

import logging
from typing import Any

import torch
from torch import nn

from cambium.exceptions import GraftingError

logger = logging.getLogger(__name__)


_DECODER_LAYER_CLASSES: dict[str, tuple[str, str]] = {
    "llama": ("transformers.models.llama.modeling_llama", "LlamaDecoderLayer"),
    "mistral": ("transformers.models.mistral.modeling_mistral", "MistralDecoderLayer"),
    "gemma": ("transformers.models.gemma.modeling_gemma", "GemmaDecoderLayer"),
    "gemma3": ("transformers.models.gemma3.modeling_gemma3", "Gemma3DecoderLayer"),
    "gemma3_text": ("transformers.models.gemma3.modeling_gemma3", "Gemma3DecoderLayer"),
    "qwen2": ("transformers.models.qwen2.modeling_qwen2", "Qwen2DecoderLayer"),
    "qwen3": ("transformers.models.qwen3.modeling_qwen3", "Qwen3DecoderLayer"),
}


def load_source_config(source_model_id: str) -> Any:
    """Load only the config of the source model (no weights)."""
    try:
        from transformers import AutoConfig
    except ImportError as e:
        raise GraftingError("transformers library required for grafting") from e

    try:
        return AutoConfig.from_pretrained(source_model_id)
    except Exception as e:
        raise GraftingError(f"Failed to load config for {source_model_id}: {e}") from e


def resolve_block_prefix(
    source_layer_attribute: str,
    source_block_idx: int | None,
    source_block_name: str | None,
) -> str:
    """Return the parameter-key prefix for the requested source block."""
    if source_block_name is not None:
        prefix = source_block_name
    elif source_block_idx is not None:
        prefix = f"{source_layer_attribute}.{source_block_idx}"
    else:
        raise GraftingError("Must provide source_block_idx or source_block_name")

    if not prefix.endswith("."):
        prefix += "."
    return prefix


def build_source_decoder_layer(config: Any, layer_idx: int) -> nn.Module:
    """Create a single decoder layer from the source config (uninitialized)."""
    model_type = getattr(config, "model_type", "llama")
    spec = _DECODER_LAYER_CLASSES.get(model_type)

    if spec is None:
        supported = ", ".join(_DECODER_LAYER_CLASSES)
        raise GraftingError(
            f"Unsupported source model type '{model_type}'. "
            f"Supported architectures: {supported}"
        )

    module_name, class_name = spec
    try:
        module = __import__(module_name, fromlist=[class_name])
        block_class = getattr(module, class_name)
    except Exception as e:
        raise GraftingError(f"Could not import decoder layer for {model_type}: {e}") from e

    try:
        return block_class(config, layer_idx=layer_idx)
    except Exception as e:
        raise GraftingError(f"Could not instantiate source block: {e}") from e


def download_block_weights(
    source_model_id: str,
    block_prefix: str,
    cache_dir: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Download only the safetensors shards that contain keys under ``block_prefix``
    and return a state dict filtered to that prefix.
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as e:
        raise GraftingError("huggingface_hub is required for selective weight download") from e

    try:
        files = list_repo_files(source_model_id, repo_type="model")
    except Exception as e:
        raise GraftingError(f"Failed to list files for {source_model_id}: {e}") from e

    safetensors_files = [f for f in files if f.endswith(".safetensors")]
    if not safetensors_files:
        raise GraftingError(
            f"No safetensors weights found in {source_model_id}. "
            "Grafting currently requires safetensors checkpoints."
        )

    index_file = "model.safetensors.index.json"
    shard_files: list[str]
    if index_file in files:
        shard_files = _shards_for_prefix(source_model_id, block_prefix, index_file, cache_dir)
    elif len(safetensors_files) == 1:
        shard_files = safetensors_files
    else:
        # Multiple shards without an index: download all to be safe.
        shard_files = safetensors_files

    return _load_filtered_weights(source_model_id, shard_files, block_prefix, cache_dir)


def _shards_for_prefix(
    source_model_id: str,
    block_prefix: str,
    index_file: str,
    cache_dir: str | None,
) -> list[str]:
    """Use the safetensors index to find shards holding the requested keys."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise GraftingError("huggingface_hub is required for selective weight download") from e

    try:
        index_path = hf_hub_download(
            source_model_id,
            filename=index_file,
            repo_type="model",
            cache_dir=cache_dir,
        )
    except Exception as e:
        raise GraftingError(f"Failed to download index for {source_model_id}: {e}") from e

    import json

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    target_shards = {shard for key, shard in weight_map.items() if key.startswith(block_prefix)}
    if not target_shards:
        raise GraftingError(f"No weights found with prefix '{block_prefix}' in {source_model_id}")

    return sorted(target_shards)


def _load_filtered_weights(
    source_model_id: str,
    shard_files: list[str],
    block_prefix: str,
    cache_dir: str | None,
) -> dict[str, torch.Tensor]:
    """Download the requested shards and return only keys matching ``block_prefix``."""
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
    except ImportError as e:
        raise GraftingError("safetensors and huggingface_hub required for grafting") from e

    filtered: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        try:
            shard_path = hf_hub_download(
                source_model_id,
                filename=shard,
                repo_type="model",
                cache_dir=cache_dir,
            )
        except Exception as e:
            raise GraftingError(f"Failed to download shard {shard}: {e}") from e

        shard_weights = load_file(shard_path)
        for key, tensor in shard_weights.items():
            if key.startswith(block_prefix):
                filtered[key] = tensor

    if not filtered:
        raise GraftingError(
            f"No weights matched prefix '{block_prefix}' after loading shards {shard_files}"
        )

    logger.info(
        f"Downloaded {len(shard_files)} shard(s) and loaded {len(filtered)} tensor(s) "
        f"for prefix '{block_prefix}'"
    )
    return filtered


def strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Remove the block prefix from state-dict keys so they match a standalone block."""
    stripped: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if not key.startswith(prefix):
            continue
        stripped[key[len(prefix) :]] = tensor
    return stripped


def load_grafted_block(
    source_model_id: str,
    source_layer_attribute: str,
    source_block_idx: int | None,
    source_block_name: str | None,
    target_dtype: torch.dtype | None = None,
    cache_dir: str | None = None,
) -> nn.Module:
    """
    Load a single decoder layer from a remote model with minimal downloads.

    Args:
        source_model_id: HuggingFace repo id of the source model.
        source_layer_attribute: Dot-separated path to the layers module in the
            source checkpoint (e.g. ``model.layers``).
        source_block_idx: Integer index of the source block.
        source_block_name: Exact layer name/prefix in the checkpoint. Overrides
            ``source_block_idx`` if provided.
        target_dtype: Optional dtype to cast loaded weights to.
        cache_dir: Optional HuggingFace cache directory.

    Returns:
        An initialized decoder layer from the source model.
    """
    config = load_source_config(source_model_id)
    prefix = resolve_block_prefix(source_layer_attribute, source_block_idx, source_block_name)

    # The standalone block uses layer_idx 0; the prefix carries the real index.
    block = build_source_decoder_layer(config, layer_idx=0)
    weights = download_block_weights(source_model_id, prefix, cache_dir)
    weights = strip_prefix(weights, prefix)

    missing, unexpected = block.load_state_dict(weights, strict=False)
    if missing:
        logger.warning(f"Source block had missing keys: {missing}")
    if unexpected:
        logger.warning(f"Source block had unexpected keys: {unexpected}")

    if target_dtype is not None:
        block = block.to(target_dtype)

    return block
