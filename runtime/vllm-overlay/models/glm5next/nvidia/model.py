# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import ClassVar, Literal

import torch
from torch import nn

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mhc import (
    MHCFusedPostPreOp,
    MHCPostOp,
    MHCPreOp,
    hc_contract,
    hc_expand,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_dequantize,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_v2 import _get_moe_router_dtype
from vllm.model_executor.models.glm4_1v import (
    Glm4vDummyInputsBuilder,
    Glm4vForConditionalGeneration,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    init_vllm_registered_model,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig

from .attention import Glm5NextMLAAttention
from .kda import Glm5NextLinearAttention
from .multimodal import (
    Glm5NextMultiModalProcessor,
    Glm5NextProcessingInfo,
    Glm5NextVisionTransformer,
)

logger = init_logger(__name__)


class Glm5NextMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel=False,
        prefix: str = "",
        swiglu_limit: float | None = None,
    ) -> None:
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )

        self.swiglu_limit = swiglu_limit
        if self.swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit=self.swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Glm5NextMoE(nn.Module):
    def __init__(
        self,
        config: Glm5NextConfig,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        apply_routed_scale_to_output: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.routed_scaling_factor = config.routed_scaling_factor

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.router_dtype = _get_moe_router_dtype(config)
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=self.router_dtype,
            prefix=f"{prefix}.gate",
        )
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32)
            )
        else:
            self.gate.e_score_correction_bias = None

        # Load balancing settings.
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        swiglu_limit = config.swiglu_limit
        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = Glm5NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
                swiglu_limit=swiglu_limit,
            )

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_token,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.moe_renormalize,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=apply_routed_scale_to_output,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=None,
            router_logits_dtype=self.gate.out_dtype,
            swiglu_limit=swiglu_limit,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape

        # Chunk the hidden states so they aren't replicated across TP ranks.
        # This avoids duplicate computation in self.experts.
        if self.is_sequence_parallel and not already_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        # The router is always external (self.gate); main's MoERunner expects
        # pre-computed router_logits, so compute them here unconditionally.
        if (
            getattr(self.experts, "gate", None) is not None
            and __import__("os").environ.get("GLM53_SKIP_DUP_ROUTER", "0") == "1"
        ):
            # the runner applies self.gate itself and overwrites whatever is passed here
            router_logits = hidden_states.new_empty(
                (num_tokens, self.gate.weight.shape[0]),
                dtype=self.gate.out_dtype or hidden_states.dtype,
            )
            if num_tokens <= 4 and _glm53_fast_router_ok(self, hidden_states):
                _glm53_fast_router(self, hidden_states)
        else:
            router_logits, _ = self.gate(hidden_states)
        nxt = getattr(self, "_glm53_next_moe", None)
        if nxt is not None and num_tokens <= 4:
            from vllm.model_executor.layers.quantization.exl3 import (
                exl3_prefetch_k,
                exl3_prefetch_slots,
                exl3_request_prefetch,
            )

            pf_k = min(exl3_prefetch_k(), exl3_prefetch_slots() // max(num_tokens, 1))
            if pf_k > 0:
                # Pre-gating: the next layer's router applied to this layer's router input
                next_logits, _ = nxt.gate(hidden_states)
                scores = next_logits.float().sigmoid()
                if nxt.gate.e_score_correction_bias is not None:
                    scores = scores + nxt.gate.e_score_correction_bias
                exl3_request_prefetch(
                    self.experts, nxt.experts, scores.topk(pf_k, dim=-1).indices
                )
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        if self.is_sequence_parallel and not already_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.view(num_tokens, hidden_dim)




_GLM53_FAST_MHC = __import__("os").environ.get("GLM53_FAST_MHC", "0").lower() in ("1", "true", "yes")
_GLM53_FAST_MHC_CTR: dict = {}


def _glm53_fast_mhc(layer, x, residual, post, comb, fn, scale, base, norm):
    """Decode mHC boundary: post (prev mixes) -> pre (new mixes) -> RMSNorm, one kernel."""
    import glm_fast

    ctr = _GLM53_FAST_MHC_CTR.get(x.device)
    if ctr is None:
        ctr = torch.zeros(1024, dtype=torch.int32, device=x.device)
        _GLM53_FAST_MHC_CTR[x.device] = ctr
    residual, post, comb, y = glm_fast.mhc_post_pre_norm(
        x.contiguous(), residual.contiguous(), post.contiguous(), comb.contiguous(),
        fn, scale, base, norm.weight.data, ctr,
        layer.rms_norm_eps, layer.hc_eps, layer.hc_eps, layer.mhc_post_mult_value,
        layer.mhc_sinkhorn_iterations, norm.variance_epsilon, 64,
    )
    return residual, post, comb, y


_GLM53_FAST_ROUTER = __import__("os").environ.get("GLM53_FAST_ROUTER", "0") == "1"
_GLM53_ROUTER_CTR: dict = {}


def _glm53_fast_router_ok(moe, x) -> bool:
    ok = getattr(moe, "_glm53_fast_router_ok", None)
    if ok is None:
        router = getattr(moe.experts, "router", None)
        w = moe.gate.weight
        ok = (
            _GLM53_FAST_ROUTER
            and router is not None
            and type(router).__name__ == "GroupedTopKRouter"
            and getattr(router, "scoring_func", None) == "sigmoid"
            and int(getattr(router, "num_expert_group", 1) or 1) == 1
            and getattr(router, "eplb_state", None) is None
            and w.dtype == torch.bfloat16
            and w.is_contiguous()
            and getattr(moe.gate, "bias", None) is None
            and w.shape[0] <= 512
            and int(router.top_k) <= 32
        )
        moe._glm53_fast_router_ok = ok
    return ok and x.dtype == torch.bfloat16 and x.is_contiguous()


def _glm53_fast_router(moe, x):
    import glm_fast

    router = moe.experts.router
    k = int(router.top_k)
    T = x.shape[0]
    qm = getattr(moe.experts, "_quant_method", None)
    idt = getattr(qm, "topk_indices_dtype", None) or torch.int32
    if idt not in (torch.int32, torch.int64):
        idt = torch.int32
    ctr = _GLM53_ROUTER_CTR.get(x.device)
    if ctr is None:
        ctr = torch.zeros(64, dtype=torch.int32, device=x.device)
        _GLM53_ROUTER_CTR[x.device] = ctr
    w = torch.empty((T, k), dtype=torch.float32, device=x.device)
    ids = torch.empty((T, k), dtype=idt, device=x.device)
    bias = router.e_score_correction_bias
    glm_fast.router_topk(
        x, moe.gate.weight, None if bias is None else bias.data, k, bool(router.renormalize),
        float(router.routed_scaling_factor), w, ids, ctr,
    )
    router._glm53_pre = (w, ids)


def _glm53_fast_mhc_ok(layer, x) -> bool:
    return (
        _GLM53_FAST_MHC
        and x.shape[0] <= 16
        and x.shape[-1] == 4096
        and not layer.is_sequence_parallel
    )


_GLM53_LAYER_DUMP_COUNTS: dict[tuple[int, str], int] = {}


def _glm53_layer_dump(layer_idx: int, tag: str, t: torch.Tensor | None) -> None:
    """Debug-only tensor dump; see patches/patch_layer_dump.py."""
    import os as _os

    if t is None or not (3 <= t.shape[0] <= 31):
        return
    if torch.cuda.is_current_stream_capturing():
        return
    if not _os.path.exists("/home/bot/llm/probe/LAYER_DUMP_ON"):
        return
    if get_tensor_model_parallel_rank() != 0:
        return
    key = (layer_idx, tag)
    n = _GLM53_LAYER_DUMP_COUNTS.get(key, 0)
    _GLM53_LAYER_DUMP_COUNTS[key] = n + 1
    _os.makedirs("/home/bot/llm/probe/layer_dump", exist_ok=True)
    torch.save(
        t.detach().cpu().clone(),
        f"/home/bot/llm/probe/layer_dump/L{layer_idx:02d}_{tag}_{n}.pt",
    )



_GLM53_ROUTE_BUF: dict[str, list] = {}
_GLM53_ROUTE_FLUSH = [0]


def _glm53_route_dump(layer_idx: int, kind: str, t: torch.Tensor) -> None:
    """Debug-only decode router-input recorder; see patches/patch_route_dump.py."""
    import os as _os

    if t.shape[0] != 1 or torch.cuda.is_current_stream_capturing():
        return
    if not _os.path.exists("/home/bot/llm/probe/ROUTE_DUMP_ON"):
        return
    if get_tensor_model_parallel_rank() != 0:
        return
    key = f"{kind}{layer_idx:02d}"
    _GLM53_ROUTE_BUF.setdefault(key, []).append(t.detach().reshape(1, -1).to(torch.bfloat16).cpu())
    if kind == "ffn" and len(_GLM53_ROUTE_BUF[key]) >= 128 and layer_idx == 44:
        _os.makedirs("/home/bot/llm/probe/route_dump", exist_ok=True)
        blob = {k: torch.cat(v) for k, v in _GLM53_ROUTE_BUF.items()}
        torch.save(blob, f"/home/bot/llm/probe/route_dump/steps_{_GLM53_ROUTE_FLUSH[0]}.pt")
        _GLM53_ROUTE_FLUSH[0] += 1
        _GLM53_ROUTE_BUF.clear()


class Glm5NextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextConfig,
        layer_idx: int,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        is_mtp_layer: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        # Official FP8 checkpoints deliberately materialize MLA projections
        # in BF16.  Unsloth EXL3 serializes q_a/kv_a/q_b/o/indexer.wq_b
        # as packed tensors, so preserve quantization for those modules.
        mla_quant_config = (
            quant_config
            if quant_config is not None
            and quant_config.get_name() == "exl3"
            else None
        )
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_moe = config.is_moe
        self.num_hidden_layers = config.num_hidden_layers
        self.rms_norm_eps = config.rms_norm_eps
        self.num_experts = config.n_routed_experts
        self.is_mtp_layer = is_mtp_layer
        self.mhc = config.mhc
        is_kda_layer = not is_mtp_layer and config.is_kda_layer(layer_idx)
        self.layer_kind = "kda" if is_kda_layer else "mla"
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if is_kda_layer:
            self.self_attn = Glm5NextLinearAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            # MLA layers require the latent head dims, which are guaranteed set
            # on MLA configs; narrow away the `int | None`.
            assert config.v_head_dim is not None
            assert config.kv_lora_rank is not None
            self.self_attn = Glm5NextMLAAttention(
                vllm_config=vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                max_position_embeddings=config.max_position_embeddings,
                cache_config=cache_config,
                quant_config=mla_quant_config,
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
                skip_rope=config.mla_nope,
            )

        # MTP layers sit past the base model's hidden layers (layer_idx >=
        # num_hidden_layers), so they're outside mlp_layer_types; default them
        # to the last base layer's MLP type (sparse/MoE for these checkpoints).
        mlp_layer_types = config.mlp_layer_types
        mlp_type = (
            mlp_layer_types[layer_idx]
            if layer_idx < len(mlp_layer_types)
            else (mlp_layer_types[-1] if mlp_layer_types else "sparse")
        )
        if self.is_moe and self.num_experts is not None and mlp_type == "sparse":
            self.mlp = Glm5NextMoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Glm5NextMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                swiglu_limit=config.swiglu_limit,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Cached for the hot forward path (isinstance per layer per step).
        self._mlp_is_moe = isinstance(self.mlp, Glm5NextMoE)
        # In SP, the attention output projection leaves a partial sum; the
        # decoder-layer reduce_scatter after attention completes it (DSv4 pattern).
        # MTP layers use the non-mHC path which has no sp_reduce_scatter, so
        # their o_proj must still reduce normally.
        if self.is_sequence_parallel and not is_mtp_layer:
            self.self_attn.o_proj.reduce_results = False
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if self.mhc and not is_mtp_layer:
            # mhc config
            self.mhc_num_residual_streams = config.mhc_num_residual_streams
            self.mhc_no_norm_weight = config.mhc_no_norm_weight
            self.mhc_tau = config.mhc_tau
            self.hc_eps = config.hc_eps
            self.mhc_sinkhorn_iterations = config.mhc_sinkhorn_iterations
            self.mhc_post_mult_value = config.mhc_post_mult_value

            n = config.mhc_num_residual_streams
            d_model = n * self.hidden_size
            mix_hc = (2 + n) * n

            self.n = n

            # attn hc
            self.hc_attn_fn = nn.Parameter(
                torch.empty(mix_hc, d_model, dtype=torch.float32)
            )
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

            # ffn hc
            self.hc_ffn_fn = nn.Parameter(
                torch.empty(mix_hc, d_model, dtype=torch.float32)
            )
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

            self.mhc_pre_op = MHCPreOp()
            self.mhc_post_op = MHCPostOp()
            self.mhc_fused_post_pre_op = MHCFusedPostPreOp()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        post: torch.Tensor | None = None,
        comb: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        # 70B or MTP layers: KDA + MoE without HC.
        if not self.mhc or self.is_mtp_layer:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            attn_output = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
            hidden_states, residual = self.post_attention_layernorm(
                attn_output, residual=residual
            )
            hidden_states = self.mlp(hidden_states)
            if self.is_mtp_layer:
                # Return the unsummed pair: the MTP caller feeds it straight
                # into shared_head's fused_add_rms_norm (one kernel instead of
                # a separate residual-add + norm). The sum itself is unchanged
                # (fp32-accumulated inside the fused kernel).
                return hidden_states, residual, None, None
            hidden_states = residual + hidden_states
            return hidden_states, residual, None, None

        # mHC start. `post`/`comb` carry the previous layer's deferred
        # hc_post inputs (its ffn-pre outputs); when present, fuse that
        # hc_post with this layer's attn hc_pre into one kernel (inter-layer
        # fusion). Layer 0 has no incoming state -> standalone hc_pre.
        x = hidden_states
        if post is None:
            if self.layer_idx == 0:
                x = hc_expand(x, self.n)
            residual = x
            if self.layer_idx == 0 and 3 <= x.shape[0] <= 31 and not torch.cuda.is_current_stream_capturing():
                import os as _os
                if _os.path.exists("/home/bot/llm/probe/DUMP_ON"):
                    _d = "/home/bot/llm/probe/kda_dump"
                    _seq = getattr(Glm5NextDecoderLayer, "_dbg_seq", 0)
                    Glm5NextDecoderLayer._dbg_seq = _seq + 1
                    torch.save(x.float().cpu(), f"{_d}/lyr0_streams_in_{_seq}.pt")
                    torch.save(residual.float().cpu(), f"{_d}/lyr0_residual_{_seq}.pt")
            post, comb, x = self.hc_pre(
                x,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=None,
                norm_eps=0.0,
            )
            # PATCH(sm89-correctness): the fused-norm tilelang kernel
            # normalizes layer_input with the flat-stream statistic
            # rsqrt(mean(x_flat^2)+eps), but reference semantics normalize
            # the COLLAPSED vector rsqrt(mean(collapsed^2)+eps). With GLM
            # tiny pre-norm activations (var ~1e-5) and eps 1e-5 this
            # shrinks layer_input per-token => gibberish. Apply the exact
            # RMSNorm here instead.
            x = self.input_layernorm(x)
        elif _glm53_fast_mhc_ok(self, x):
            residual, post, comb, x = _glm53_fast_mhc(
                self, x, residual, post, comb, self.hc_attn_fn, self.hc_attn_scale,
                self.hc_attn_base, self.input_layernorm,
            )
        else:
            residual, post, comb, x = self.hc_fused_post_pre(
                x,
                residual,
                post,
                comb,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=None,
                norm_eps=0.0,
            )
            x = self.input_layernorm(x)  # PATCH(sm89-correctness): see above

        # Attention needs the full token sequence; mHC above ran on the SP
        # shard. Gather for attention, scatter back afterward (DSv4 pattern).
        if self.is_sequence_parallel:
            x = sp_all_gather(x)[: positions.shape[0]]

        if self.layer_idx == 0 and 3 <= x.shape[0] <= 31 and not torch.cuda.is_current_stream_capturing():
            import os as _os
            if _os.path.exists("/home/bot/llm/probe/DUMP_ON"):
                _seq = getattr(Glm5NextDecoderLayer, "_dbg_seq", 1) - 1
                torch.save(x.float().cpu(), f"/home/bot/llm/probe/kda_dump/lyr0_pre_attn_x_{_seq}.pt")
                if _seq == 2 or not hasattr(self, "_w_dumped"):
                    self._w_dumped = True
                    _d = "/home/bot/llm/probe/kda_dump"
                    torch.save(self.hc_attn_fn.float().cpu(), f"{_d}/mdl_fn.pt")
                    torch.save(self.hc_attn_base.float().cpu(), f"{_d}/mdl_base.pt")
                    torch.save(self.hc_attn_scale.float().cpu(), f"{_d}/mdl_scale.pt")
                    torch.save(self.input_layernorm.weight.data.float().cpu(), f"{_d}/mdl_normw.pt")
                    from vllm.model_executor.layers.mhc import _apply_mhc_norm

                    _p, _c, _x = self.mhc_pre_op.forward_native(
                        residual,
                        self.hc_attn_fn,
                        self.hc_attn_scale,
                        self.hc_attn_base,
                        rms_eps=self.rms_norm_eps,
                        hc_pre_eps=self.hc_eps,
                        hc_sinkhorn_eps=self.hc_eps,
                        hc_post_mult_value=self.mhc_post_mult_value,
                        sinkhorn_repeat=self.mhc_sinkhorn_iterations,
                    )
                    _x = _apply_mhc_norm(_x, self.input_layernorm.weight.data, self.input_layernorm.variance_epsilon)
                    torch.save(_x.float().cpu(), f"{_d}/lyr0_native_x.pt")
                    torch.save(x.float().cpu(), f"{_d}/lyr0_tile_x.pt")
        _glm53_layer_dump(self.layer_idx, "attn_in", x)
        _glm53_route_dump(self.layer_idx, "att", x)
        x = self.self_attn(
            hidden_states=x,
            positions=positions,
        )
        _glm53_layer_dump(self.layer_idx, "attn_out", x)
        _tib = getattr(self.self_attn, "topk_indices_buffer", None)
        if _tib is None:
            _tib = getattr(getattr(self.self_attn, "indexer", None), "topk_indices_buffer", None)
        if _tib is not None:
            _glm53_layer_dump(self.layer_idx, "topk", _tib[: x.shape[0]])

        if self.is_sequence_parallel:
            x = sp_reduce_scatter(x)

        # Fuse post-attn hc_post + pre-FFN hc_pre (+ RMSNorm) into one kernel.
        if _glm53_fast_mhc_ok(self, x):
            residual, post, comb, x = _glm53_fast_mhc(
                self, x, residual, post, comb, self.hc_ffn_fn, self.hc_ffn_scale,
                self.hc_ffn_base, self.post_attention_layernorm,
            )
        else:
            residual, post, comb, x = self.hc_fused_post_pre(
                x,
                residual,
                post,
                comb,
                self.hc_ffn_fn,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                norm_weight=None,
                norm_eps=0.0,
            )
            x = self.post_attention_layernorm(x)  # PATCH(sm89-correctness): see above

        # Fully Connected
        _glm53_layer_dump(self.layer_idx, "mlp_in", x)
        _glm53_route_dump(self.layer_idx, "ffn", x)
        if self._mlp_is_moe:
            x = self.mlp(x, already_sequence_parallel=self.is_sequence_parallel)
        else:
            x = self.mlp(x)
        _glm53_layer_dump(self.layer_idx, "mlp_out", x)

        # mHC end. The last mHC layer materializes its final hc_post (nothing
        # to fuse with) then contracts; every other layer defers its hc_post to
        # the next layer's fused pre, returning the state.
        if self.layer_idx == self.num_hidden_layers - 1:
            x = self.hc_post(x, residual, post, comb)
            x = hc_contract(x, self.n)
            return x, None, None, None

        return x, residual, post, comb

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ):
        post_mix, res_mix, layer_input = self.mhc_pre_op(
            residual=x,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.mhc_post_mult_value,
            sinkhorn_repeat=self.mhc_sinkhorn_iterations,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
        return post_mix, res_mix, layer_input

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        return self.mhc_post_op(x, residual, post, comb)

    def hc_fused_post_pre(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ):
        return self.mhc_fused_post_pre_op(
            x=x,
            residual=residual,
            post_layer_mix=post,
            comb_res_mix=comb,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.mhc_post_mult_value,
            sinkhorn_repeat=self.mhc_sinkhorn_iterations,
            n_splits=1,
            tile_n=1,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )


class Glm5NextModel(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config

        self.vocab_size = config.vocab_size
        self.device = current_platform.device_type

        """
        if config.index_topk is not None:
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=self.device,
            )
        else:
        """
        self.is_v32 = config.index_topk is not None
        if self.is_v32:
            topk_tokens = config.index_topk
            assert topk_tokens is not None
            # Reserve room for the incomplete pool tail.
            kpool = config.index_kpool
            assert kpool is not None
            buffer_width = topk_tokens + (kpool - 1 if kpool > 1 else 0)
            # Sparse MLA tiles top-k in 128 columns; padded slots remain masked.
            sparse_topk_block_n = 128
            buffer_width = (
                (buffer_width + sparse_topk_block_n - 1) // sparse_topk_block_n
            ) * sparse_topk_block_n
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                buffer_width,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            # Full-MLA config (no kpool sparse indexer): no topk buffer.
            topk_indices_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return Glm5NextDecoderLayer(
                vllm_config=vllm_config,
                config=config,
                layer_idx=layer_idx,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        # The active slice is fixed after construction; cache it so forward
        # doesn't rebuild the slice (a fresh list) every step.
        self._active_layers = self.layers[self.start_layer : self.end_layer]
        # Link each routed MoE to the next one (cross-layer expert prefetch, EXL3_PREFETCH_K)
        _moes = [
            layer.mlp
            for layer in self._active_layers
            if isinstance(getattr(layer, "mlp", None), Glm5NextMoE)
        ]
        for _a, _b in zip(_moes, _moes[1:]):
            object.__setattr__(_a, "_glm53_next_moe", _b)

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.is_sequence_parallel = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )

        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0, (
            "num_attention_heads must be divisible by world_size"
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        n = self.config.mhc_num_residual_streams
        hidden_size = self.config.hidden_size
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, hidden_size), dtype=dtype, device=device
                ),
                "residual": torch.zeros(
                    (batch_size, n, hidden_size), dtype=dtype, device=device
                ),
                "post": torch.zeros(
                    (batch_size, n, 1), dtype=torch.float32, device=device
                ),
                "comb": torch.zeros(
                    (batch_size, n, n), dtype=torch.float32, device=device
                ),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
            post = None
            comb = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            post = intermediate_tensors["post"]
            comb = intermediate_tensors["comb"]

        full_num_tokens = positions.shape[0]
        if self.is_sequence_parallel:
            hidden_states = sp_shard(hidden_states)

        import os as _os
        _flag = "/home/bot/llm/probe/DUMP_ON"
        _dump_dir = "/home/bot/llm/probe/vllm_dump" if _os.path.exists(_flag) else None
        import torch as _torch

        _dump_on = (
            _dump_dir
            and get_tensor_model_parallel_rank() == 0
            and not _torch.cuda.is_current_stream_capturing()
            and 3 <= positions.shape[0] <= 31
            and getattr(Glm5NextModel, "_glm53_dump_seq", 0)
            < 12
        )
        if _dump_on and input_ids is not None and not bool(
            input_ids.abs().sum().item()
        ):
            _dump_on = False
        if _dump_on:
            _seq = getattr(Glm5NextModel, "_glm53_dump_seq", 0)
            Glm5NextModel._glm53_dump_seq = _seq + 1
            _sub = f"{_dump_dir}/vllm_seq{_seq}"
            _os.makedirs(_sub, exist_ok=True)
            _torch.save(
                {
                    "input_ids": input_ids.cpu() if input_ids is not None else None,
                    "positions": positions.cpu(),
                },
                f"{_sub}/meta.pt",
            )
        # DFlash2 consumes the outputs of five target layers. GLM's mHC path
        # defers each mid-stack hc_post to the next layer, so materialize a
        # side-effect-free reconstruction before contracting the four streams.
        aux_hidden_states: list[torch.Tensor] = []
        for _li, layer in enumerate(self._active_layers):
            hidden_states, residual, post, comb = layer(
                positions, hidden_states, residual, post, comb
            )
            if _os.environ.get("VLLM_GLM53_LAYER_SYNC") == "1" and not _torch.cuda.is_current_stream_capturing():
                try:
                    _torch.cuda.synchronize()
                except Exception:
                    import logging as _lg
                    _lg.getLogger("vllm").error(
                        "LAYER_SYNC fault after layer %d tokens=%d pos=[%d..%d]",
                        self.start_layer + _li, positions.shape[-1],
                        int(positions.min().item()) if False else -1, -1,
                    )
                    raise
            if _dump_on:
                _torch.save(hidden_states.float().cpu(), f"{_sub}/x_{_li:02d}.pt")
                if residual is not None:
                    _torch.save(
                        residual.float().cpu(), f"{_sub}/r_{_li:02d}.pt"
                    )
            if self.start_layer + _li + 1 in self.aux_hidden_state_layers:
                if post is not None:
                    aux_value = hc_contract(
                        layer.hc_post(hidden_states, residual, post, comb),
                        layer.n,
                    )
                else:
                    aux_value = hidden_states
                    if aux_value.ndim == 3:
                        aux_value = aux_value.mean(dim=1)
                if self.is_sequence_parallel:
                    aux_value = sp_all_gather(aux_value)[:full_num_tokens]
                aux_hidden_states.append(aux_value)

        if not get_pp_group().is_last_rank:
            assert residual is not None and post is not None and comb is not None
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                    "post": post,
                    "comb": comb,
                }
            )

        if self.is_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]

        hidden_states = self.norm(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            # MLA: fuse q_a_proj and kv_a_proj_with_mqa
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            # Indexer: fuse wk and weights_proj
            (".wk_weights_proj", ".wk", 0),
            (".wk_weights_proj", ".weights_proj", 1),
        ]
        exl3_kda = (
            self.quant_config is not None
            and self.quant_config.get_name() == "exl3"
        )
        if exl3_kda:
            # qkv_proj is already fused in the EXL3 checkpoint and loads by
            # exact name; only the native b/f_a/g_a tensors need remapping.
            stacked_params_mapping.extend(
                [
                    (".in_proj_bfg_a", ".b_proj", 0),
                    (".in_proj_bfg_a", ".f_a_proj", 1),
                    (".in_proj_bfg_a", ".g_a_proj", 2),
                ]
            )
        else:
            stacked_params_mapping.extend(
                [
                    (".in_proj_qkvbfg_a", ".q_proj", 0),
                    (".in_proj_qkvbfg_a", ".k_proj", 1),
                    (".in_proj_qkvbfg_a", ".v_proj", 2),
                    (".in_proj_qkvbfg_a", ".b_proj", 3),
                    (".in_proj_qkvbfg_a", ".f_a_proj", 4),
                    (".in_proj_qkvbfg_a", ".g_a_proj", 5),
                ]
            )
        if self.config.is_moe:
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
            expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.n_routed_experts,
            )
        else:
            expert_params_mapping = []
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # GLM-5.3-Flash NoPE checkpoints omit the RoPE rows from
        # ``kv_a_proj_with_mqa``; pad them with zeros for the model shape.
        kv_a_pad_size = 0
        if self.config.mla_nope and self.config.qk_rope_head_dim > 0:
            kv_a_pad_size = self.config.qk_rope_head_dim

        _pending_wk_fp8: dict = {}

        for args in weights:
            name, loaded_weight = args[:2]
            # Skip sparse-indexer tensors only when the indexer module itself
            # was disabled (index_topk=None via hf_overrides).  Checking the
            # raw checkpoint name against params_dict here is incorrect:
            # wk.weight and weights_proj.weight are intentionally remapped
            # below into the fused wk_weights_proj parameter, so their raw
            # names cannot be present in params_dict.  The old check silently
            # skipped both learned projections and left the sparse indexer at
            # its zero/uninitialised constructor values.
            if ".indexer." in name and self.config.index_topk is None:
                continue
            kwargs: dict = args[2] if len(args) > 2 else {}
            # The EXL3 checkpoint stores the KDA depthwise q/k/v convolution fused
            # along output channels, while this GLM implementation keeps three
            # ColumnParallelLinear parameters.  Split first, then let each
            # parameter's normal loader apply the TP output-channel shard.
            if exl3_kda and ".conv1d.weight" in name and name not in params_dict:
                if loaded_weight.shape[0] % 3:
                    raise ValueError(
                        f"Fused KDA conv1d has non-divisible shape {loaded_weight.shape}"
                    )
                for conv_label, conv_weight in zip(
                    ("q", "k", "v"), loaded_weight.chunk(3, dim=0)
                ):
                    conv_name = name.replace(
                        ".conv1d.weight", f".{conv_label}_conv1d.weight"
                    )
                    if is_pp_missing_parameter(conv_name, self):
                        continue
                    conv_param = params_dict[conv_name]
                    conv_param.weight_loader(conv_param, conv_weight)
                    loaded_params.add(conv_name)
                continue
            if "rotary_emb.inv_freq" in name:
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue

            # Handle FP8 indexer WK: dequantize to BF16 for fusion with
            # weights_proj into wk_weights_proj.
            if _try_load_fp8_indexer_wk(
                name,
                loaded_weight,
                _pending_wk_fp8,
                params_dict,
                loaded_params,
            ):
                continue

            # FP8 checkpoint: dequantize BF16-kept MLA projections
            # (q_a_proj / kv_a_proj_with_mqa / o_proj) to BF16.
            if _try_load_fp8_attn_proj(
                name,
                loaded_weight,
                _pending_wk_fp8,
                params_dict,
                loaded_params,
                kv_a_pad_size,
            ):
                continue

            # Pad kv_a_proj_with_mqa for NoPE models
            if kv_a_pad_size > 0 and ".kv_a_proj_with_mqa." in name:
                pad = torch.zeros(
                    kv_a_pad_size,
                    *loaded_weight.shape[1:],
                    dtype=loaded_weight.dtype,
                    device=loaded_weight.device,
                )
                loaded_weight = torch.cat([loaded_weight, pad], dim=0)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                # QKV fusion: skip if fused module doesn't exist in model
                if param_name == ".fused_qkv_a_proj" and name_mapped not in params_dict:
                    continue
                name = name_mapped
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for idx, (
                    param_name,
                    weight_name,
                    expert_id,
                    expert_shard_id,
                ) in enumerate(expert_params_mapping):
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=expert_shard_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias")
                        and name not in params_dict
                        and not self.config.is_linear_attn
                    ):  # noqa: E501
                        continue
                    # PATCH(sm89-longctx-v2): when the kpool sparse indexer is
                    # disabled (index_topk=None via hf_overrides), the module is
                    # not built, but quantized checkpoints still carry indexer
                    # tensors. Skip them instead of raising KeyError.
                    if ".indexer." in name and name not in params_dict:
                        continue
                    # Remapping the name of FP8 kv-scale.
                    remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped_name is None:
                        continue
                    name = remapped_name
                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, **kwargs)
            loaded_params.add(name)
        import os as _os
        if (
            _os.environ.get("VLLM_GLM53_EMBED_UVA") == "1"
            and isinstance(self.embed_tokens, VocabParallelEmbedding)
            and not getattr(self.embed_tokens.weight, "_vllm_is_uva_offloaded", False)
        ):
            from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

            w = self.embed_tokens.weight
            host = torch.empty(w.shape, dtype=w.dtype, device="cpu", pin_memory=True)
            host.copy_(w.data)
            w.data = get_accelerator_view_from_cpu_tensor(host)
            w._vllm_is_uva_offloaded = True
            self._glm53_embed_host = host
            torch.cuda.empty_cache()
            logger.info("GLM53 embed_tokens moved to UVA host memory: %.1f MiB",
                        host.numel() * host.element_size() / 2**20)
        return loaded_params


class Glm5NextForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsPP,
    MixtureOfExperts,
    IsHybrid,
    SupportsEagle3,
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config = self.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        self.model = Glm5NextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, scale=self.config.logit_scale
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        return self.model.make_empty_intermediate_tensors(batch_size, dtype, device)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.kda_state_shape(
            tp_size,
            hf_config.linear_num_heads,
            hf_config.linear_head_dim,
            conv_kernel_size=hf_config.linear_conv_kernel_dim,
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[
        MambaStateCopyFunc, MambaStateCopyFunc, MambaStateCopyFunc, MambaStateCopyFunc
    ]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


@MULTIMODAL_REGISTRY.register_processor(
    Glm5NextMultiModalProcessor,
    info=Glm5NextProcessingInfo,
    dummy_inputs=Glm4vDummyInputsBuilder,
)
class Glm5NextForConditionalGeneration(
    Glm4vForConditionalGeneration, HasInnerState, IsHybrid, SupportsEagle3
):
    # The text model (KDA + dense-MLA + MoE) is a hybrid mamba model. The
    # multimodal wrapper must declare the same interfaces so vLLM treats it as
    # hybrid (auto-aligns mamba/attention block sizes, sizes the mamba state
    # cache); the mamba-state classmethods delegate to the text model.
    has_inner_state: ClassVar[Literal[True]] = True
    is_hybrid: ClassVar[Literal[True]] = True

    # NOTE: weight-prefix mapping is inherited from Glm4vForConditionalGeneration
    # (``model.visual.`` -> ``visual.``, ``model.language_model.`` ->
    # ``language_model.model.``, ``lm_head.`` -> ``language_model.lm_head.``),
    # matching the GLM-OCR / GLM-4V serialization convention. If the real
    # checkpoint's safetensors keys differ (e.g. ``language_model.model.`` with
    # no outer ``model.``), override ``hf_to_vllm_mapper`` accordingly.

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        from .model import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        from .model import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        from .model import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Glm4vForConditionalGeneration, self).__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        assert multimodal_config is not None

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Glm5NextVisionTransformer(
                config.text_config,
                config.vision_config,
                # Read eps from the VISION sub-config, not the top-level
                # `config.rms_norm_eps`: Glm5NextConfig.__getattribute__ mirrors
                # the latter onto text_config (1e-5), silently ignoring the
                # vision tower's own (1e-6) rms_norm_eps.
                norm_eps=config.vision_config.rms_norm_eps,
                # Vision tower ships BF16 weights in this fp8 checkpoint (no
                # weight_scale_inv for visual.*), so it must NOT inherit the
                # global fp8 quant_config -- doing so incorrectly quantizes
                # the tower
                # and yields NaN image features. Mirrors the MLA/KDA proj
                # pattern (quant_config=None for BF16 submodules).
                quant_config=None,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Glm5NextForCausalLM"],
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def get_encoder_cudagraph_config(self):
        # This vision tower does not produce the absolute position embedding
        # buffer used by GLM4V.
        config = super().get_encoder_cudagraph_config()
        config.buffer_keys = [k for k in config.buffer_keys if k != "pos_embeds"]
        return config


def get_spec_layer_idx_from_weight_name(
    config: Glm5NextConfig, weight_name: str
) -> int | None:
    if hasattr(config, "num_nextn_predict_layers") and (
        config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(
                f"model.layers.{layer_idx + i}."
            ) or weight_name.startswith(f"layers.{layer_idx + i}."):
                return layer_idx + i
    return None


def _try_load_fp8_indexer_wk(name, tensor, buf, params_dict, loaded_params):
    if "indexer.wk." not in name or "wk_weights" in name:
        return False
    is_weight = name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn
    is_scale = "weight_scale_inv" in name
    if not is_weight and not is_scale:
        return False
    layer_prefix = name.rsplit(".wk.", 1)[0]
    entry = buf.setdefault(layer_prefix, {})
    entry["weight" if is_weight else "scale"] = tensor
    if "weight" not in entry or "scale" not in entry:
        return True

    weight_fp8, scale_inv = entry["weight"], entry["scale"]
    del buf[layer_prefix]
    block_size = weight_fp8.shape[1] // scale_inv.shape[1]
    weight_bf16 = scaled_dequantize(
        weight_fp8,
        scale_inv,
        group_shape=GroupShape(block_size, block_size),
        out_dtype=torch.bfloat16,
    )

    fused_name = f"{layer_prefix}.wk_weights_proj.weight"
    param = params_dict[fused_name]
    param.weight_loader(param, weight_bf16, 0)
    loaded_params.add(fused_name)
    return True


def _dequant_fp8_block(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Dequantize a block-FP8 (e4m3) weight with per-block scale to BF16.

    Unlike ``scaled_dequantize`` this tolerates a non-divisible (partial last
    block) shape by zero-padding to a multiple of ``block_size`` before the
    scale broadcast and trimming back afterwards (e.g. kv_a_proj_with_mqa is
    576 rows = 4*128 + 64).
    """
    out_dim, in_dim = weight_fp8.shape
    pad_out = (-out_dim) % block_size
    pad_in = (-in_dim) % block_size
    w = weight_fp8
    if pad_out or pad_in:
        w = torch.nn.functional.pad(w, (0, pad_in, 0, pad_out))
    # scale_inv is (ceil(out/block), ceil(in/block)); broadcast to (out, in).
    s = scale_inv.to(torch.float32)
    s_full = s.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    out = (w.to(torch.float32) * s_full).to(torch.bfloat16)
    return out[:out_dim, :in_dim].contiguous()


# FP8 checkpoint projections that the MODEL keeps in BF16, so the block-FP8
# (weight + weight_scale_inv) must be dequantized to BF16 on load.
# Maps checkpoint proj-suffix -> (buffer key, model target base, fused shard id
# or None for a direct projection, whether NoPE rope-padding applies).
_FP8_ATTN_PROJS = {
    ".q_a_proj.": ("q_a", "fused_qkv_a_proj", 0, False),
    ".kv_a_proj_with_mqa.": ("kv_a", "fused_qkv_a_proj", 1, True),
    ".q_b_proj.": ("q_b", "q_b_proj", None, False),
    ".o_proj.": ("o_proj", "o_proj", None, False),
}


def _try_load_fp8_attn_proj(
    name,
    tensor,
    buf,
    params_dict,
    loaded_params,
    kv_a_pad_size: int,
) -> bool:
    """Dequantize FP8 q_a_proj / kv_a_proj_with_mqa / o_proj to BF16 on load.

    The FP8 checkpoint stores these as block-FP8 (weight + weight_scale_inv),
    but the model holds them in BF16 (``fused_qkv_a_proj`` is always BF16 via
    DeepSeekV2FusedQkvAProjLinear; ``o_proj`` is excluded by
    modules_to_not_convert). When the model target is BF16 (no
    ``weight_scale_inv`` param) we dequantize; otherwise we return False so the
    normal stacked/direct path loads the FP8 tensor as-is.
    """
    matched = None
    for suffix, info in _FP8_ATTN_PROJS.items():
        if suffix in name:
            matched = (suffix, info)
            break
    if matched is None:
        return False
    suffix, (key, target_base, shard_id, is_kva) = matched
    is_weight = name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn
    is_scale = "weight_scale_inv" in name
    if not is_weight and not is_scale:
        return False

    layer_prefix = name.rsplit(suffix, 1)[0]
    target_w = f"{layer_prefix}.{target_base}.weight"
    target_s = f"{layer_prefix}.{target_base}.weight_scale_inv"
    # If the model actually kept this projection in FP8, let the normal path
    # handle it (it has a weight_scale_inv param).
    if target_s in params_dict:
        return False

    entry = buf.setdefault(layer_prefix, {}).setdefault(key, {})
    entry["weight" if is_weight else "scale"] = tensor
    if "weight" not in entry or "scale" not in entry:
        return True

    weight_fp8, scale_inv = entry["weight"], entry["scale"]
    buf[layer_prefix].pop(key, None)
    block_size = weight_fp8.shape[1] // scale_inv.shape[1]
    weight_bf16 = _dequant_fp8_block(weight_fp8, scale_inv, block_size)
    # NoPE: pad kv_a rope portion (kv_lora_rank -> kv_lora_rank + qk_rope_head_dim).
    if is_kva and kv_a_pad_size > 0:
        pad = torch.zeros(
            kv_a_pad_size,
            weight_bf16.shape[1],
            dtype=weight_bf16.dtype,
            device=weight_bf16.device,
        )
        weight_bf16 = torch.cat([weight_bf16, pad], dim=0)

    param = params_dict[target_w]
    if shard_id is None:
        param.weight_loader(param, weight_bf16)
    else:
        param.weight_loader(param, weight_bf16, shard_id)
    loaded_params.add(target_w)
    return True
