from __future__ import annotations

import io
import sys
import types

import torch

import freetoken.models.qwen4_exp.gguf_experts as gguf_experts
from freetoken.models.gguf.dequant import GGML_Q8_0, row_bytes


def test_expert_loader_reads_payloads_without_touching_gguf_tensor_data(monkeypatch) -> None:
    experts = 2
    hidden = intermediate = 32
    half = intermediate * row_bytes(hidden, GGML_Q8_0)
    down = hidden * row_bytes(intermediate, GGML_Q8_0)

    class Tensor:
        def __init__(self, name: str, offset: int, n_bytes: int):
            self.name = name
            self.tensor_type = GGML_Q8_0
            self.data_offset = offset
            self.n_bytes = n_bytes

        @property
        def data(self):
            raise AssertionError("expert loader must not fault GGUF tensor.data")

    gate_bytes = bytes([11]) * (experts * half)
    up_bytes = bytes([22]) * (experts * half)
    down_bytes = bytes([33]) * (experts * down)
    payload = gate_bytes + up_bytes + down_bytes

    class Reader:
        def __init__(self, _path: str):
            self.tensors = [
                Tensor("blk.0.ffn_gate_exps.weight", 0, len(gate_bytes)),
                Tensor("blk.0.ffn_up_exps.weight", len(gate_bytes), len(up_bytes)),
                Tensor("blk.0.ffn_down_exps.weight", len(gate_bytes) + len(up_bytes), len(down_bytes)),
            ]

    class HostBank:
        def __init__(self, shape, dtype):
            self.tensor = torch.zeros(shape, dtype=dtype)

    config = types.SimpleNamespace(
        num_moe_layers=1,
        num_experts=experts,
        hidden_size=hidden,
        moe_intermediate_size=intermediate,
        qwen4_args=types.SimpleNamespace(gguf_expert_types=[(GGML_Q8_0, GGML_Q8_0)]),
    )

    import freetoken.distributed as distributed
    import freetoken.models.gguf.reader as reader_module
    import freetoken.moe.host_banks as host_banks

    monkeypatch.setattr(distributed, "get_tp_info", lambda: types.SimpleNamespace(size=1))
    monkeypatch.setattr(reader_module, "gguf_config_source", lambda _: "shard-1")
    monkeypatch.setattr(reader_module, "gguf_split_paths", lambda _: ("shard-1",))
    monkeypatch.setattr(host_banks, "HostBank", HostBank)
    monkeypatch.setattr(host_banks, "pin_banks", lambda _: None)
    monkeypatch.setattr("builtins.open", lambda *_args, **_kwargs: io.BytesIO(payload))
    monkeypatch.setitem(sys.modules, "gguf", types.SimpleNamespace(GGUFReader=Reader))

    banks = gguf_experts.load_gguf_expert_sources("unused.gguf", config)

    assert torch.equal(banks["gate_up"][0][:, :half], torch.full((experts, half), 11, dtype=torch.uint8))
    assert torch.equal(banks["gate_up"][0][:, half : 2 * half], torch.full((experts, half), 22, dtype=torch.uint8))
    assert torch.equal(banks["down"][0][:, :down], torch.full((experts, down), 33, dtype=torch.uint8))
