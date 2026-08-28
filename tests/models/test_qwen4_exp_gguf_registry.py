from freetoken.models.gguf.config import GGUF_ARCH_TO_REGISTRY
from freetoken.models.register import get_model_spec


def test_qwen4exp_gguf_architecture_routes_to_native_runtime():
    key = GGUF_ARCH_TO_REGISTRY["qwen4exp"]
    assert key == "Qwen4ExpGGUFForCausalLM"
    spec = get_model_spec(key)
    assert spec.module == "freetoken.models.qwen4_exp"
    assert spec.model_cls == "Qwen4ExpGGUFForCausalLM"
    assert spec.parse_config == "parse_gguf_config"
    assert spec.iter_weights == "iter_gguf_weights"
