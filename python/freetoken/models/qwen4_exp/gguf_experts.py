"""Routed-expert host banks for Qwen4Exp Unsloth/llama.cpp GGUF checkpoints.

The published UD quants are heterogeneous by decoder layer. Host RAM therefore
keeps every layer at its native packed width while FreeToken's #199 geometry
cache allocates GPU arenas from the distinct row geometries.

Only routed expert payloads are materialized here. In particular, the huge PLE
embedding tensor is never converted to a contiguous NumPy/Torch buffer by this
loader.
"""

from __future__ import annotations

import numpy as np
import torch

from freetoken.models.gguf.dequant import GGML_NAME, row_bytes

_EXPERT_SUFFIXES = (
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
)


def _require_tp1() -> None:
    from freetoken.distributed import get_tp_info

    if get_tp_info().size != 1:
        raise NotImplementedError(
            "Qwen4Exp GGUF routed expert banks currently support TP=1 only"
        )


def _geometry(config) -> list[tuple[int, int]]:
    """Compact 64-byte-aligned ``(gate_up, down)`` bytes/expert per layer."""
    types = getattr(config.qwen4_args, "gguf_expert_types", None)
    if not types:
        raise ValueError("Qwen4Exp GGUF config has no routed expert tensor types")
    if len(types) != config.num_moe_layers:
        raise ValueError(
            f"Qwen4Exp GGUF has {len(types)} expert type entries, "
            f"expected {config.num_moe_layers}"
        )

    H, I = config.hidden_size, config.moe_intermediate_size

    def align64(n: int) -> int:
        return (n + 63) // 64 * 64

    return [
        (
            align64(2 * I * row_bytes(H, gate_up_type)),
            align64(H * row_bytes(I, down_type)),
        )
        for gate_up_type, down_type in types
    ]


def _packed_tensor_bytes(tensor) -> torch.Tensor:
    """Materialize one selected GGUF tensor as a flat uint8 CPU tensor."""
    raw = np.ascontiguousarray(tensor.data).reshape(-1).view(np.uint8)
    return torch.from_numpy(raw)


def load_gguf_expert_sources(
    model_path: str, config, *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    """Load compact native packed routed-expert banks.

    Returns two banks, ``gate_up`` and ``down``. Each is a list with one
    ``[num_experts, bytes_per_expert]`` uint8 HostBank tensor per decoder layer.
    Gate and up are fused inside each expert row in that order. Different layers
    may have different row widths; #199's geometry cache handles that on GPU.
    """
    import gguf

    from freetoken.models.gguf.reader import gguf_config_source, gguf_split_paths
    from freetoken.moe.host_banks import HostBank, pin_banks

    _require_tp1()
    if layer_sink is not None:
        # Direct serving is the first milestone. Reject before allocating ~50 GB
        # instead of silently retaining it when a converter expected streaming.
        raise NotImplementedError("Qwen4Exp GGUF FTW streaming conversion is not wired yet")

    types = config.qwen4_args.gguf_expert_types
    if not types:
        raise ValueError("Qwen4Exp GGUF expert types were not recovered from the tensor table")

    source = gguf_config_source(model_path)
    if source is None:
        raise ValueError(f"cannot resolve Qwen4Exp GGUF source from {model_path!r}")

    L = config.num_moe_layers
    E = config.num_experts
    H = config.hidden_size
    I = config.moe_intermediate_size
    geometry = _geometry(config)

    hb = {
        "gate_up": [HostBank((E, gu_stride), torch.uint8) for gu_stride, _ in geometry],
        "down": [HostBank((E, dn_stride), torch.uint8) for _, dn_stride in geometry],
    }
    banks = {name: [bank.tensor for bank in per_layer] for name, per_layer in hb.items()}

    gate_up_parts: dict[int, dict[str, torch.Tensor]] = {}
    seen_gate: set[int] = set()
    seen_up: set[int] = set()
    seen_down: set[int] = set()

    for path in gguf_split_paths(source):
        reader = gguf.GGUFReader(path)
        for tensor in reader.tensors:
            name = tensor.name
            if not name.startswith("blk.") or not name.endswith(_EXPERT_SUFFIXES):
                continue

            _, layer_text, suffix = name.split(".", 2)
            layer = int(layer_text)
            if not (0 <= layer < L):
                raise ValueError(f"{name}: expert layer {layer} outside [0, {L})")

            gate_up_type, down_type = types[layer]
            actual_type = int(tensor.tensor_type)

            if suffix == "ffn_down_exps.weight":
                if layer in seen_down:
                    raise ValueError(f"duplicate routed expert tensor {name}")
                if actual_type != down_type:
                    raise ValueError(
                        f"{name}: header type changed from {GGML_NAME.get(down_type, down_type)} "
                        f"to {GGML_NAME.get(actual_type, actual_type)}"
                    )
                payload = H * row_bytes(I, down_type)
                src = _packed_tensor_bytes(tensor).reshape(E, payload)
                banks["down"][layer][:, :payload].copy_(src)
                seen_down.add(layer)
                continue

            if actual_type != gate_up_type:
                raise ValueError(
                    f"{name}: header type changed from "
                    f"{GGML_NAME.get(gate_up_type, gate_up_type)} to "
                    f"{GGML_NAME.get(actual_type, actual_type)}"
                )

            half = I * row_bytes(H, gate_up_type)
            src = _packed_tensor_bytes(tensor).reshape(E, half)
            part_name = "gate" if suffix == "ffn_gate_exps.weight" else "up"
            parts = gate_up_parts.setdefault(layer, {})
            if part_name in parts:
                raise ValueError(f"duplicate routed expert tensor {name}")
            parts[part_name] = src
            if part_name == "gate":
                seen_gate.add(layer)
            else:
                seen_up.add(layer)

            if "gate" in parts and "up" in parts:
                dst = banks["gate_up"][layer]
                dst[:, :half].copy_(parts["gate"])
                dst[:, half : 2 * half].copy_(parts["up"])
                del gate_up_parts[layer]

    if gate_up_parts:
        details = {layer: sorted(parts) for layer, parts in gate_up_parts.items()}
        raise RuntimeError(f"incomplete gate/up expert layers: {details}")

    want = set(range(L))
    missing_gate = sorted(want - seen_gate)
    missing_up = sorted(want - seen_up)
    missing_down = sorted(want - seen_down)
    if missing_gate or missing_up or missing_down:
        raise RuntimeError(
            "Qwen4Exp GGUF is missing routed expert tensors: "
            f"gate={missing_gate}, up={missing_up}, down={missing_down}"
        )

    if torch.cuda.is_available():
        pin_banks(hb)
    return banks


def dummy_gguf_expert_sources(config) -> dict[str, list[torch.Tensor]]:
    """Fabricate native-byte banks with the same heterogeneous layer geometry."""
    from freetoken.moe.host_banks import HostBank, pin_banks

    _require_tp1()
    E = config.num_experts
    geometry = _geometry(config)
    hb = {
        "gate_up": [HostBank((E, gu_stride), torch.uint8) for gu_stride, _ in geometry],
        "down": [HostBank((E, dn_stride), torch.uint8) for _, dn_stride in geometry],
    }
    banks = {name: [bank.tensor for bank in per_layer] for name, per_layer in hb.items()}
    for tensor in banks["gate_up"] + banks["down"]:
        tensor.random_(0, 256)
    if torch.cuda.is_available():
        pin_banks(hb)
    return banks


__all__ = ["load_gguf_expert_sources", "dummy_gguf_expert_sources"]
