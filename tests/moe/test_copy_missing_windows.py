"""``_copy_missing_windows`` is the pure-PyTorch Windows fallback for the tvm-ffi
fused miss-copy kernels (which fail to compile under MSVC). It must move exactly
the same bytes as the paths it replaces:

* the legacy unified-cache miss copy (state on the cache),
* the geometry-pool decode miss copy (state lives on the POOL, targets are the
  pool's per-bank views -- reading ``self.*`` here would silently copy nothing),
* the zero-miss no-op,
* the whole-layer geometry prefill.

Runs on CPU so it stays platform-independent: the fallback touches no CUDA JIT.
"""

from __future__ import annotations

import torch

from freetoken.moe.offload_cache import OffloadMoeCache, _GeometryPoolState

SENTINEL = 0xEE
NUM_LAYERS, NUM_EXPERTS, CACHE_SIZE = 2, 6, 10
ROW_WIDTHS = {0: 8, 1: 16}  # heterogeneous per-layer rows -> flat max-stride arena


def _layer_tensor(layer: int, width: int) -> torch.Tensor:
    experts = torch.arange(NUM_EXPERTS)[:, None]
    cols = torch.arange(width)[None, :]
    return ((layer * 100 + experts + cols) % 251).to(torch.uint8)


def _build_cache():
    cache = OffloadMoeCache(
        num_layers=NUM_LAYERS,
        num_experts=NUM_EXPERTS,
        cache_size=CACHE_SIZE,
        device=torch.device("cpu"),
        cache_policy="lru",
        prefill_overlap=False,
        quant_format="gguf",
    )
    sources = {
        "gate_up": [_layer_tensor(layer, w) for layer, w in ROW_WIDTHS.items()],
        "down": [
            (layer * 50 + w) * torch.ones((NUM_EXPERTS, w), dtype=torch.uint8)
            for layer, w in ROW_WIDTHS.items()
        ],
    }
    cache.set_bank_sources(sources)
    for _, c in cache.banks:
        c.fill_(SENTINEL)
    return cache, sources


def _attach_pool(cache, layer_ids=(1,), slots=4, row_bytes=(16, 16), offset=0):
    """Layer-1 geometry pool carved from the arena exactly like _init_geometry_pools."""
    views = []
    for name, row in zip(cache.bank_schema, row_bytes):
        arena = cache.bank_caches[name].reshape(-1)
        views.append(arena.narrow(0, offset, slots * row).view(slots, row))
        offset += slots * row
    return _GeometryPoolState(
        num_layers=cache.num_layers,
        num_experts=cache.num_experts,
        cache_size=slots,
        device=cache.device,
        layer_ids=layer_ids,
        row_bytes=tuple(row_bytes),
        bank_views=tuple(views),
    )


def test_legacy_miss_copy_moves_exact_rows_only():
    cache, sources = _build_cache()
    dst = torch.tensor([0, 5, 9], dtype=torch.int32)
    src_rows = torch.tensor([5, 1, 4], dtype=torch.int32)
    cache.num_indices.fill_(dst.numel())
    cache.evict_slots[: dst.numel()] = dst
    cache.src_indices[: dst.numel()] = src_rows
    cache._pending_src_layer = 0  # narrow source rows: arena tails must survive
    cache._pending_geometry_pool = None
    cache._pending_geometry_prefill = False

    cache._copy_missing_windows(0)

    width = ROW_WIDTHS[0]
    for name in cache.bank_schema:
        arena = cache.bank_caches[name]
        for j, (slot, expert) in enumerate(zip(dst.tolist(), src_rows.tolist())):
            assert torch.equal(arena[slot, :width], sources[name][0][expert])
            assert (arena[slot, width:] == SENTINEL).all()
        untouched = torch.ones(CACHE_SIZE, dtype=torch.bool)
        untouched[dst.long()] = False
        assert (arena[untouched] == SENTINEL).all()


def test_geometry_pool_miss_copy_uses_pool_state_and_views():
    cache, sources = _build_cache()
    pool = _attach_pool(cache)
    dst = torch.tensor([0, 2, 3], dtype=torch.int32)
    src_rows = torch.tensor([5, 1, 4], dtype=torch.int32)
    pool.num_indices.fill_(dst.numel())
    pool.evict_slots[: dst.numel()] = dst
    pool.src_indices[: dst.numel()] = src_rows
    # stale main-cache state the fallback must NOT read when a pool is pending
    cache.num_indices.zero_()
    cache._pending_src_layer = 1
    cache._pending_geometry_pool = pool
    cache._pending_geometry_prefill = False

    cache._copy_missing_windows(1)

    for name, view in zip(cache.bank_schema, pool.bank_views):
        for j, (slot, expert) in enumerate(zip(dst.tolist(), src_rows.tolist())):
            assert torch.equal(view[slot], sources[name][1][expert])
        untouched = torch.ones(pool.cache_size, dtype=torch.bool)
        untouched[dst.long()] = False
        assert (view[untouched] == SENTINEL).all()
    # rows beyond the pool's arena carve-out stay untouched
    assert (cache.bank_caches["gate_up"][pool.cache_size :] == SENTINEL).all()


def test_zero_miss_is_a_no_op():
    cache, _ = _build_cache()
    pool = _attach_pool(cache)
    cache.num_indices.zero_()
    cache._pending_src_layer = 1
    cache._pending_geometry_pool = pool
    cache._pending_geometry_prefill = False

    cache._copy_missing_windows(1)

    for name, view in zip(cache.bank_schema, pool.bank_views):
        assert (view == SENTINEL).all(), name


def test_geometry_prefill_copies_whole_layer_into_arena():
    cache, sources = _build_cache()
    cache._pending_src_layer = 0
    cache._pending_geometry_prefill = True

    cache._copy_missing_windows(0)

    width = ROW_WIDTHS[0]
    for name in cache.bank_schema:
        arena = cache.bank_caches[name]
        assert torch.equal(arena[:NUM_EXPERTS, :width], sources[name][0])
        assert (arena[:NUM_EXPERTS, width:] == SENTINEL).all()
        assert (arena[NUM_EXPERTS:] == SENTINEL).all()
