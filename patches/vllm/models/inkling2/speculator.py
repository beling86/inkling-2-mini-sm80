# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator for the Inkling2 draft: stores the raw target features
(each draft layer mixes them itself) and runs its recurrent Markov head."""

import os

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.platforms import current_platform
from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import (
    _DRAFT_NOISE_SALT,
    _gumbel_combine_kernel,
    _log1p_neg_stable,
    murmur3_uniform32,
    spec_step_seed,
)
from vllm.v1.worker.gpu.spec_decode.draft_logits import draft_logits_row
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

try:
    from inkling_sm80_cuda import HAS_CUDA_EXT as _SM80_HAS_CUDA
    from inkling_sm80_cuda import (
        fused_markov_fold_gumbel_b1_cuda as _sm80_fused_markov_fold_gumbel_b1,
    )
except Exception:
    _SM80_HAS_CUDA = False
    _sm80_fused_markov_fold_gumbel_b1 = None


@triton.jit
def _sm80_add_softcap_gumbel_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    logits_cache_ptr,
    lc_s0,
    lc_s1,
    lc_col_ptr,
    lc_row_map_ptr,
    base_ptr,
    base_stride,
    bias_ptr,
    bias_stride,
    idx_map_ptr,
    idx_map_stride,
    seeds_ptr,
    pos_ptr,
    pos_stride,
    temp_ptr,
    vocab_size,
    SOFTCAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_CACHE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    base = tl.load(
        base_ptr + token_idx * base_stride + block, mask=mask, other=0.0
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + token_idx * bias_stride + block, mask=mask, other=0.0
    ).to(tl.float32)
    s = (base + bias).to(tl.bfloat16)
    s_cap = (tldevice.tanh(s / SOFTCAP) * SOFTCAP).to(tl.bfloat16)

    req_idx = tl.load(idx_map_ptr + token_idx * idx_map_stride).to(tl.int64)
    is_valid = req_idx >= 0
    temp = tl.load(temp_ptr + req_idx, mask=is_valid, other=0.0).to(tl.float32)
    seed = tl.load(seeds_ptr + req_idx, mask=is_valid, other=0)
    pos = tl.load(pos_ptr + token_idx * pos_stride) - 1

    if HAS_CACHE:
        col = tl.load(lc_col_ptr)
        row = draft_logits_row(req_idx, lc_row_map_ptr)
        tl.store(
            logits_cache_ptr + row * lc_s0 + col * lc_s1 + block,
            s_cap,
            mask=mask & (row >= 0),
        )
        seed = spec_step_seed(seed, col)

    logits = s_cap.to(tl.float32)
    if temp != 0.0:
        logits = logits / temp
        u = murmur3_uniform32(seed, pos + _DRAFT_NOISE_SALT, block.to(tl.uint32))
        logits = logits + (-tl.log(-_log1p_neg_stable(u)))
    logits = tl.where(mask, logits, float("-inf"))
    lmax, lidx = tl.max(logits, axis=0, return_indices=True)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, lmax)
    tl.store(
        local_argmax_ptr + token_idx * local_argmax_stride + block_idx,
        (block_idx * BLOCK_SIZE + lidx).to(tl.int64),
    )


class Inkling2DSparkSpeculator(DSparkSpeculator):
    _speculator_name = "Inkling2 DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        config = self.draft_model_config.hf_config
        assert not self.sample_from_anchor
        assert self._draft_topk is None and not self.enable_adaptive_verification
        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            len(config.target_layer_ids) * config.hidden_size,
            dtype=self.dtype,
            device=device,
        )
        # combine_hidden_states is the identity: store features eagerly.
        self._aux_staging = None
        # Read by prepare_context_anchor inside the FULL draft graph.
        self._num_rejected = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.f_rnn = torch.zeros(
            self.max_num_reqs, config.hidden_size, dtype=self.dtype, device=device
        )

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # Draft base logits come from a bf16 GEMM, unlike the target's fp32 head.
        return self.dtype, 0.0

    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
    ) -> nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        model.attach_target(target_model.get_language_model())
        return model

    def prepare_context_anchor(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        num_rejected: torch.Tensor | None,
    ) -> None:
        assert num_rejected is not None
        ends = query_start_loc[1 : num_reqs + 1] - num_rejected[:num_reqs]
        rows = self.hidden_states[(ends - 1).clamp_min(0).long()]
        self.f_rnn[:num_reqs] = self.model.compute_f_rnn(rows)

    def _sm80_can_fuse_softcap_gumbel(self) -> bool:
        return (
            current_platform.is_device_capability_family(80)
            and os.environ.get("INKLING_SM80_NO_FUSED_GUMBEL") != "1"
            and get_tensor_model_parallel_world_size() == 1
            and self.draft_logits is not None
            and self.draft_watermarker is None
            and self._d2t_scatter_index is None
            and self.acceptance_estimator is None
            and not self.use_fp64_gumbel
            and not self.use_confidence_head
        )

    def _sm80_fused_softcap_gumbel(
        self,
        base: torch.Tensor,
        bias: torch.Tensor,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        assert self.draft_logits is not None
        num_reqs = base.shape[0]
        vocab_size = self.model.config.vocab_size
        softcap = float(self.model.config.final_logit_softcapping)
        block_size = 2048 if num_reqs <= 2 else 1024
        num_warps = 4 if num_reqs <= 2 else 8
        num_blocks = triton.cdiv(vocab_size, block_size)
        local_argmax = torch.empty(
            (num_reqs, num_blocks), dtype=torch.int64, device=base.device
        )
        local_max = torch.empty(
            (num_reqs, num_blocks), dtype=torch.float32, device=base.device
        )
        out_tok = torch.empty((num_reqs,), dtype=torch.int64, device=base.device)
        cache_vals = self.draft_logits.values
        _sm80_add_softcap_gumbel_kernel[(num_reqs, num_blocks)](
            local_argmax,
            local_argmax.stride(0),
            local_max,
            local_max.stride(0),
            cache_vals,
            cache_vals.stride(0),
            cache_vals.stride(1),
            self._step_cols[step],
            self.draft_logits.row_mapping,
            base,
            base.stride(0),
            bias,
            bias.stride(0),
            idx_map,
            idx_map.stride(0),
            self.seeds,
            sample_pos,
            sample_pos.stride(0),
            self.temperature,
            vocab_size,
            SOFTCAP=softcap,
            BLOCK_SIZE=block_size,
            HAS_CACHE=True,
            num_warps=num_warps,
        )
        _gumbel_combine_kernel[(num_reqs,)](
            local_argmax,
            local_argmax.stride(0),
            local_max,
            local_max.stride(0),
            out_tok,
            num_blocks,
            PADDED_NUM_BLOCKS=triton.next_power_of_2(num_blocks),
        )
        return out_tok

    def _sm80_fused_markov_fold_gumbel_b1(
        self,
        latent: torch.Tensor,
        base: torch.Tensor,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        assert self.draft_logits is not None
        mh = self.model.model.markov_head
        assert mh.fold_weight is not None and mh.fold_bias is not None
        vocab_size = self.model.config.vocab_size
        softcap = float(self.model.config.final_logit_softcapping)
        block_v = 256  # 8 warps x 32 rows
        num_blocks = triton.cdiv(vocab_size, block_v)
        local_argmax = torch.empty((1, num_blocks), dtype=torch.int64, device=base.device)
        local_max = torch.empty((1, num_blocks), dtype=torch.float32, device=base.device)
        out_tok = torch.empty((1,), dtype=torch.int64, device=base.device)
        _sm80_fused_markov_fold_gumbel_b1(
            latent,
            mh.fold_weight,
            mh.fold_bias,
            base,
            self.draft_logits.values,
            local_argmax,
            local_max,
            idx_map,
            self.seeds,
            sample_pos,
            self.temperature,
            self._step_cols[step],
            self.draft_logits.row_mapping,
            vocab_size,
            softcap,
        )
        _gumbel_combine_kernel[(1,)](
            local_argmax,
            local_argmax.stride(0),
            local_max,
            local_max.stride(0),
            out_tok,
            num_blocks,
            PADDED_NUM_BLOCKS=triton.next_power_of_2(num_blocks),
        )
        return out_tok

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_base_logits(hidden).view(num_reqs, n_spec, -1)
        context = self.model.model.markov_head.context(hidden, self.f_rnn[:num_reqs])
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        state = hidden.new_zeros(num_reqs, self.model.config.markov_rank)
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        use_fused = self._sm80_can_fuse_softcap_gumbel()
        mh = self.model.model.markov_head
        use_fused_b1 = (
            use_fused
            and _SM80_HAS_CUDA
            and os.environ.get("INKLING_SM80_NO_CUDA_MFG") != "1"
            and num_reqs == 1
            and mh.markov_rank == 256
            and mh.fold_weight is not None
            and mh.fold_bias is not None
            and mh.fold_weight.is_contiguous()
            and mh.fold_bias.is_contiguous()
        )
        for i in range(n_spec):
            if use_fused_b1:
                state, latent = mh.step_latent(state, prev, context[i])
                prev = self._sm80_fused_markov_fold_gumbel_b1(
                    latent, base_logits[:, i], idx_map[:, i], sample_pos[:, i], i
                )
            else:
                state, bias = self.model.markov_step(state, prev, context[i])
                if self.draft_logits is None:
                    prev = self.model.markov_argmax(base_logits[:, i], bias)
                elif use_fused:
                    prev = self._sm80_fused_softcap_gumbel(
                        base_logits[:, i], bias, idx_map[:, i], sample_pos[:, i], i
                    )
                else:
                    logits = self.model.markov_logits(base_logits[:, i], bias)
                    prev = self._sample_logits(logits, idx_map[:, i], sample_pos[:, i], i)
            self.draft_tokens[:num_reqs, i] = prev
