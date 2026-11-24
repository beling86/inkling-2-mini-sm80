"""Stand-ins for compiled vLLM ops that vllm/vllm-openai:inkling-2-mini ships without sm_80 kernels (170HX), 21 Nov 2026.

The image's _C / _C_cache_ops / _moe_C libraries carry sm_100 code for nearly everything (on sm_80 only Marlin and
some MoE kernels exist), so such an op launches nothing on a 170HX and leaves cudaErrorNoKernelImageForDevice behind.
The image's vLLM commit is not public, so the libraries cannot be rebuilt; instead each op the Inkling 2 Mini path
reaches is re-registered here for the CUDA dispatch key, so every caller (eager, torch.compile, CUDA graphs) gets it.
Found with INKLING_SM80_TRACE=1 (sitecustomize): on start-up and decode only reshape_and_cache_flash (the DSpark
drafter's KV write) was hit; moe_align_block_size is replaced in the patched moe_sm90.py instead.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _reshape_and_cache_flash_kernel(K, V, KC, VC, S, k_t, k_h, v_t, v_h, kc_b, kc_p, kc_h, vc_b, vc_p, vc_h,
                                    BLOCK_SIZE: tl.constexpr, D: tl.constexpr, DP: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    slot = tl.load(S + t).to(tl.int64)
    b = slot // BLOCK_SIZE
    p = slot % BLOCK_SIZE
    d = tl.arange(0, DP)
    m = (d < D) & (slot >= 0)
    k = tl.load(K + t * k_t + h * k_h + d, mask=m)
    v = tl.load(V + t * v_t + h * v_h + d, mask=m)
    tl.store(KC + b * kc_b + p * kc_p + h * kc_h + d, k, mask=m)
    tl.store(VC + b * vc_b + p * vc_p + h * vc_h + d, v, mask=m)


def reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale):
    # key/value [T, H, D] (T may be padded past slot_mapping), caches [blocks, block_size, H, D] (strided views).
    if kv_cache_dtype not in ("auto", "bfloat16", "float16") or key_cache.dtype != key.dtype:
        raise NotImplementedError(f"sm_80 reshape_and_cache_flash: kv cache dtype {kv_cache_dtype} / {key_cache.dtype}")
    n = slot_mapping.shape[0]
    if n == 0:
        return
    heads, dim = key.shape[1], key.shape[2]
    assert key.stride(2) == 1 and value.stride(2) == 1 and key_cache.stride(3) == 1 and value_cache.stride(3) == 1
    _reshape_and_cache_flash_kernel[(n, heads)](
        key, value, key_cache, value_cache, slot_mapping,
        key.stride(0), key.stride(1), value.stride(0), value.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        BLOCK_SIZE=key_cache.shape[1], D=dim, DP=triton.next_power_of_2(dim), num_warps=1)


@triton.jit
def _vocab_parallel_embedding_kernel(
    Out,
    Ids,
    W,
    out_t,
    w_v,
    org_start,
    org_end,
    add_start,
    add_end,
    add_offset,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0)
    d_blk = tl.program_id(1)
    tok = tl.load(Ids + t).to(tl.int64)
    in_org = (tok >= org_start) & (tok < org_end)
    in_add = (tok >= add_start) & (tok < add_end)
    valid = in_org | in_add
    row = tl.where(in_org, tok - org_start, tl.where(in_add, tok - add_offset, 0))
    cols = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
    cmask = cols < D
    vals = tl.load(W + row * w_v + cols, mask=valid & cmask, other=0.0)
    tl.store(Out + t * out_t + cols, vals, mask=cmask)


def vocab_parallel_embedding(
    out,
    input_ids,
    weight,
    org_vocab_start_index,
    org_vocab_end_index,
    num_org_vocab_padding,
    added_vocab_start_index,
    added_vocab_end_index,
):
    n = input_ids.shape[0]
    if n == 0:
        return
    d = weight.shape[1]
    assert out.stride(1) == 1 and weight.stride(1) == 1
    add_offset = (
        int(added_vocab_start_index)
        - (int(org_vocab_end_index) - int(org_vocab_start_index))
        - int(num_org_vocab_padding)
    )
    block_d = 256
    _vocab_parallel_embedding_kernel[(n, triton.cdiv(d, block_d))](
        out,
        input_ids,
        weight,
        out.stride(0),
        weight.stride(0),
        int(org_vocab_start_index),
        int(org_vocab_end_index),
        int(added_vocab_start_index),
        int(added_vocab_end_index),
        add_offset,
        D=d,
        BLOCK_D=block_d,
        num_warps=4,
    )


_LIBS = []


def install():
    lib = torch.library.Library("_C_cache_ops", "IMPL")
    lib.impl("reshape_and_cache_flash", reshape_and_cache_flash, "CUDA")
    _LIBS.append(lib)  # keep the registration alive
    lib_c = torch.library.Library("_C", "IMPL")
    lib_c.impl("vocab_parallel_embedding", vocab_parallel_embedding, "CUDA")
    _LIBS.append(lib_c)
    print("[sm80] reshape_and_cache_flash + vocab_parallel_embedding -> Triton stand-ins", flush=True)
