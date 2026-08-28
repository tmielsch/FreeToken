from __future__ import annotations

import transformers
from transformers.integrations import ggml

import freetoken.models.gguf.tokenizer as gguf_tokenizer


def test_qwen4exp_gguf_routes_to_qwen3_tokenizer_converter(monkeypatch) -> None:
    meta = {
        "tokenizer.ggml.tokens": ["<unk>", "<bos>", "<eos>", "hello"],
        "tokenizer.ggml.bos_token_id": 1,
        "tokenizer.ggml.eos_token_id": 2,
        "tokenizer.ggml.unknown_token_id": 0,
        "tokenizer.ggml.padding_token_id": 0,
    }
    converter_arch: list[str] = []

    class FakeTokenizer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def convert(architecture, tokenizer_dict):
        converter_arch.append(architecture)
        assert tokenizer_dict["bos_token"] == "<bos>"
        assert tokenizer_dict["eos_token"] == "<eos>"
        return object(), None

    monkeypatch.setattr(gguf_tokenizer, "load_gguf_metadata", lambda _: meta)
    monkeypatch.setattr(gguf_tokenizer, "gguf_architecture", lambda _: "qwen4exp")
    monkeypatch.setattr(ggml, "convert_gguf_tokenizer", convert)
    monkeypatch.setattr(transformers, "PreTrainedTokenizerFast", FakeTokenizer)

    tokenizer = gguf_tokenizer.load_gguf_tokenizer("qwen38.gguf")

    assert converter_arch == ["qwen3"]
    assert tokenizer.kwargs["eos_token"] == "<eos>"
