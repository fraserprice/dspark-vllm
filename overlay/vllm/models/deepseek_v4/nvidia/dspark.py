# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import typing
from collections.abc import Callable, Iterable, Mapping

import regex as re
import torch
import torch.nn as nn

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_post_tilelang,
)
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.deepseek_v4.common.ops import mtp_shared_head_rmsnorm

from .model import (
    DeepseekV4DecoderLayer,
    make_deepseek_v4_expert_params_mapping,
)

logger = init_logger(__name__)

DSPARK_ARCH: str = "DSparkDraftModel"

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


class DSparkDeepseekV4Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.hc_dim = self.hc_mult * self.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        num_hidden_layers = config.num_hidden_layers
        num_draft_blocks = len(config.compress_ratios) - num_hidden_layers
        combine_in = len(config.dspark_target_layer_ids) * self.hidden_size

        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )
        self.aux_stream_list = [torch.cuda.Stream() for _ in range(3)]

        self.embed_tokens: VocabParallelEmbedding | None = None

        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(
                    vllm_config,
                    f"{prefix}.layers.{num_hidden_layers + i}",
                    topk_indices_buffer=self.topk_indices_buffer,
                    aux_stream_list=self.aux_stream_list,
                )
                for i in range(num_draft_blocks)
            ]
        )

        self.main_proj = ReplicatedLinear(
            combine_in,
            self.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.main_proj",
        )
        self.main_norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )

        self.logits_processor = LogitsProcessor(config.vocab_size)

    @property
    def sliding_attention_layer_names(self) -> set[str]:
        return {layer.attn.swa_cache_layer.prefix for layer in self.layers}

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed_tokens is not None
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        needs_squeeze = hidden_states.dim() == 1
        hidden_states = (
            hidden_states.unsqueeze(0) if needs_squeeze else hidden_states
        )
        result = self.main_proj(hidden_states)
        return result.squeeze(0) if needs_squeeze else result

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        assert context_positions.dtype == torch.int64
        num_ctx = context_states.shape[0]
        normed = torch.empty_like(context_states)
        ops.rms_norm(
            normed,
            context_states,
            self.main_norm.weight.data,
            self.main_norm.variance_epsilon,
        )

        for layer in self.layers:
            attn = layer.attn
            qr_kv, _ = attn.fused_wqa_wkv(normed)
            kv = qr_kv[:, attn.q_lora_rank :].contiguous()
            kv_normed = torch.empty_like(kv)
            ops.rms_norm(kv_normed, kv, attn.kv_norm.weight.data, attn.eps)

            if context_slot_mapping is None:
                continue

            layer_slot_mapping = (
                context_slot_mapping[attn.swa_cache_layer.prefix]
                if isinstance(context_slot_mapping, Mapping)
                else context_slot_mapping
            )
            swa_kv_cache = attn.swa_cache_layer.kv_cache
            swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)
            dummy_q = torch.empty(
                (num_ctx, attn.n_local_heads, attn.head_dim),
                dtype=kv_normed.dtype,
                device=kv_normed.device,
            )
            torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                dummy_q,
                kv_normed,
                swa_kv_cache_2d,
                layer_slot_mapping,
                context_positions,
                attn.rotary_emb.cos_sin_cache,
                attn.padded_heads,
                attn.eps,
                attn.swa_cache_layer.block_size,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else self.embed_input_ids(input_ids)
        )
        hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)

        residual: torch.Tensor | None = None
        post_mix: torch.Tensor | None = None
        res_mix: torch.Tensor | None = None
        for layer in self.layers:
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                None,
                post_mix,
                res_mix,
                residual,
            )

        assert residual is not None
        assert post_mix is not None
        assert res_mix is not None
        last_layer = self.layers[-1]
        if last_layer._should_run_b12x_mhc(int(hidden_states.shape[0])):
            from b12x.integration.residual import b12x_mhc_post

            hidden_states = b12x_mhc_post(
                hidden_states, residual, post_mix, res_mix
            )
        else:
            hidden_states = mhc_post_tilelang(
                hidden_states, residual, post_mix, res_mix
            )
        return hidden_states.flatten(1)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
    ) -> torch.Tensor | None:
        hidden_states = hidden_states.view(-1, self.hc_mult, self.hidden_size)
        hidden_states = hc_head_fused_kernel_tilelang(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        hidden_states = mtp_shared_head_rmsnorm(
            hidden_states,
            self.norm.weight.data,
            self.norm.variance_epsilon,
        )
        return self.logits_processor(lm_head, hidden_states)


class DSparkDraftModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config

        self.model = DSparkDeepseekV4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head: nn.Module | None = None

        dtype = vllm_config.model_config.dtype
        self.markov_w1 = nn.Parameter(
            torch.empty(config.vocab_size, config.dspark_markov_rank, dtype=dtype),
            requires_grad=False,
        )
        self.markov_w2 = nn.Parameter(
            torch.empty(config.vocab_size, config.dspark_markov_rank, dtype=dtype),
            requires_grad=False,
        )
        self.confidence_proj: nn.Parameter | None = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        assert self.lm_head is not None
        return self.model.compute_logits(hidden_states, self.lm_head)

    @property
    def sliding_attention_layer_names(self) -> set[str]:
        return self.model.sliding_attention_layer_names

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        config = self.config
        num_hidden_layers = config.num_hidden_layers

        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        first_layer = self.model.layers[0]
        expert_mapping = (
            make_deepseek_v4_expert_params_mapping(config.n_routed_experts)
            if first_layer.ffn.use_mega_moe
            else fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=config.n_routed_experts,
            )
        )
        expert_scale_suffix = (
            ".weight_scale"
            if getattr(config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv"
        )

        for name, loaded_weight in weights:
            if not name.startswith("mtp."):
                continue
            parts = name.split(".", 2)
            mtp_idx = int(parts[1])
            rest = parts[2]

            if rest.startswith("markov_head.markov_w1"):
                self.markov_w1.data.copy_(loaded_weight)
                loaded_params.add("markov_w1")
                continue
            if rest.startswith("markov_head.markov_w2"):
                self.markov_w2.data.copy_(loaded_weight)
                loaded_params.add("markov_w2")
                continue
            if rest.startswith("confidence_head.proj"):
                self.confidence_proj = nn.Parameter(
                    loaded_weight.to(device=self.markov_w1.device),
                    requires_grad=False,
                )
                loaded_params.add("confidence_proj")
                continue

            if (
                rest.startswith("main_proj")
                or rest.startswith("main_norm")
                or rest.startswith("norm.")
                or rest.startswith("hc_head_")
            ):
                name = "model." + rest
            else:
                name = f"model.layers.{mtp_idx}." + rest

            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else ".weight_scale_inv"
                )
                name = name.removesuffix(".scale") + suffix

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        param = params_dict[name_mapped]
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            loaded_params.add(name_mapped)
                            break
                    continue
                elif "attn_sink" in name:
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                else:
                    if ".shared_experts.w2" in name:
                        name = name.replace(
                            ".shared_experts.w2", ".shared_experts.down_proj"
                        )
                    if name.endswith(".ffn.gate.bias"):
                        name = name.replace(
                            ".ffn.gate.bias",
                            ".ffn.gate.e_score_correction_bias",
                        )
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue

        for layer in self.model.layers:
            layer.ffn.finalize_mega_moe_weights()
            layer.refresh_b12x_mhc_bf16_weights()

        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params
