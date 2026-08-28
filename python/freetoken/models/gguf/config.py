"""GGUF config shim: the object the model registry sees for a ``.gguf`` model.

``cached_load_hf_config`` returns one of these for GGUF paths instead of a HF
``PretrainedConfig``. It carries the architecture key (so the registry can dispatch),
the raw GGUF metadata dict, and a few derived facts that need the tensor table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .reader import (
    _reader,
    gguf_architecture,
    gguf_split_paths,
    gguf_tensor_names,
    load_gguf_metadata,
)

GGUF_ARCH_TO_REGISTRY: dict[str, str] = {
    "gemma4": "Gemma4GGUFForCausalLM",
    "qwen4exp": "Qwen4ExpGGUFForCausalLM",
}


@dataclass(frozen=True)
class GgufConfigShim:
    architectures: list[str]
    model_path: str
    model_type: str
    metadata: dict[str, Any]
    vocab_size: int
    tie_word_embeddings: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "architectures": list(self.architectures),
            "model_type": self.model_type,
            "torch_dtype": "bfloat16",
            "vocab_size": self.vocab_size,
            "tie_word_embeddings": self.tie_word_embeddings,
        }


def _vocab_size(model_path: str) -> int:
    # Header-only: never touch the ~90 GB Unsloth payload just to size the vocab.
    for path in gguf_split_paths(model_path):
        for tensor in _reader(path).tensors:
            if tensor.name == "token_embd.weight":
                # ggml tensor shape is [hidden, vocab].
                return int(tensor.shape[-1])
    toks = load_gguf_metadata(model_path).get("tokenizer.ggml.tokens")
    if toks is not None:
        return len(toks)
    raise ValueError(f"GGUF {model_path}: no token_embd.weight to size the vocab")


def build_gguf_shim(model_path: str) -> GgufConfigShim:
    arch = gguf_architecture(model_path)
    registry_key = GGUF_ARCH_TO_REGISTRY.get(arch)
    if registry_key is None:
        raise ValueError(
            f"GGUF architecture {arch!r} is not supported "
            f"(known: {sorted(GGUF_ARCH_TO_REGISTRY)})"
        )
    names = gguf_tensor_names(model_path)
    metadata = load_gguf_metadata(model_path)
    if names:
        tie_word_embeddings = "output.weight" not in names
    else:
        from .reader import OUTPUT_WEIGHT_PRESENT_KV

        present = metadata.get(OUTPUT_WEIGHT_PRESENT_KV)
        if present is None:
            raise ValueError(
                f"{model_path}: metadata-only GGUF lacks {OUTPUT_WEIGHT_PRESENT_KV!r}; "
                "reconvert the checkpoint with the current freetoken.checkpoint.convert"
            )
        tie_word_embeddings = not present
    return GgufConfigShim(
        architectures=[registry_key],
        model_path=model_path,
        model_type=arch,
        metadata=metadata,
        vocab_size=_vocab_size(model_path),
        tie_word_embeddings=tie_word_embeddings,
    )


__all__ = ["GgufConfigShim", "GGUF_ARCH_TO_REGISTRY", "build_gguf_shim"]
