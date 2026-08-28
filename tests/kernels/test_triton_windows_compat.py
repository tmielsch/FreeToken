from freetoken.kernel.triton.windows_compat import patch_cuda_utils_source


def test_patch_cuda_utils_source_replaces_only_invalid_empty_initializers() -> None:
    source = """
CUlaunchAttribute clusterAttr = {};
CUlaunchAttribute clusterSchedulingAttr = {};
CUlaunchAttribute unrelated = {1};
"""

    patched = patch_cuda_utils_source(source)

    assert "clusterAttr = {0};" in patched
    assert "clusterSchedulingAttr = {0};" in patched
    assert "unrelated = {1};" in patched
    assert patch_cuda_utils_source(patched) == patched
