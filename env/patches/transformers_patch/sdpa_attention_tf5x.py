import torch

from ..utils import is_torch_npu_available, is_torch_xpu_available, logging
from ..utils.import_utils import is_torch_greater_or_equal


logger = logging.get_logger(__name__)


_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
_is_torch_greater_or_equal_than_2_8 = is_torch_greater_or_equal("2.8", accept_dev=True)
_is_torch_xpu_available = is_torch_xpu_available()
_is_torch_npu_available = is_torch_npu_available()


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def use_gqa_in_sdpa(attention_mask: torch.Tensor | None, key: torch.Tensor) -> bool:
    # GQA can only be used under the following conditions
    # 1.cuda or Ascend NPU
    #   - torch version >= 2.5
    #   - attention_mask is None (otherwise it will fall back to the math kernel)
    # 2.xpu
    #   - torch version >= 2.8
    if _is_torch_xpu_available:
        return _is_torch_greater_or_equal_than_2_8
    return _is_torch_greater_or_equal_than_2_5 and attention_mask is None


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # --- Saliency-R1 identity trick (ported to transformers 5.x) ---
    # Stock SDPA never materialises softmax(QK^T), so it returns (attn_output, None) and
    # `output_attentions=True` is unsupported. The record-outputs hook system in transformers
    # 5.x captures a registered attention module's 2nd return value (`output[1]`) IF it is not
    # None (utils/output_capturing.py). By calling SDPA a second time with V replaced by the
    # identity matrix -- softmax(QK^T) @ I == softmax(QK^T) -- we recover the attention-
    # probability matrix while keeping the fast fused kernel, and return it as `attn_weight`
    # so the hook can record it. This file is the ONLY patch Qwen3-VL needs for saliency
    # attention extraction (the model-forward plumbing is handled generically by 5.x).
    # NB: `attn_weight` is computed unconditionally -- an extra SDPA + a [b, h, kv, kv]
    # identity per call -- so only enable it for the saliency reward pass, not full training.
    if kwargs.get("output_attentions", False):
        # NOTE: unlike stock SDPA, this patched op DOES return real attention weights
        # (via the identity trick below), so `output_attentions=True` is supported here.
        logger.warning_once(
            "Saliency-R1 patch: `sdpa` is returning attention weights via the identity trick"
            " (extra SDPA + [b, h, kv, kv] identity per call). Enable only for the saliency"
            " reward pass, not full training."
        )
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)

    # SDPA's Flash Attention (and cuDNN) kernels rely on the `is_causal` flag. However, there are certain conditions:
    # - Not in decoding phase (otherwise we want full attention on the single query token)
    # - Attention mask is not to be provided (even if it is a causal pattern)
    # - Internally, we marked this as compatible with causal, i.e. it is a decoder attention type
    #
    # Quirks on the conditionals:
    # - We avoid inline passing this to the SDPA function directly to support both torch.compile's dynamic shapes and
    #   full graph options. Otherwise, dynamic shapes are prevented from compiling.
    # - It is important to check first for the shape, otherwise compile will fail with
    #   `argument 'is_causal' must be bool, not SymBool`.
    is_causal = query.shape[2] > 1 and attention_mask is None and is_causal

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # When `is_causal = False` and the `attention_mask` is not of boolean type, the Ascend NPU's SDPA interface cannot utilize the FlashAttentionScore operator，
    # and falls back to small-operator concatenation. To invoke the FlashAttentionScore, the attention_mask must be converted to boolean type.
    # This adaptation ensures the `attention_mask` meets the requirement for using FlashAttentionScore.
    if _is_torch_npu_available:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            # Convert to boolean type, making sdpa to force call FlashAttentionScore to improve performance.
            attention_mask = torch.logical_not(attention_mask.bool()).to(query.device)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        **sdpa_kwargs,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    # --- identity trick: recover softmax(QK^T) as `attn_weight` ---
    # `value` here is the (possibly repeat_kv'd) V; its dim-2 is the key/value sequence length.
    # Building the identity from `value.shape` keeps the same head count as the main call, so
    # `enable_gqa` (if set) broadcasts KV heads identically for both calls.
    batch_size, num_heads, seq_len, _ = value.shape
    identity = torch.eye(seq_len, seq_len, dtype=query.dtype, device=query.device)  # [seq_len, seq_len]
    identity = identity[None, None].expand(batch_size, num_heads, seq_len, seq_len).contiguous()
    attn_weight = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        identity,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        **sdpa_kwargs,
    ).contiguous()  # [batch, num_heads, q_len, kv_len] == softmax(QK^T)

    return attn_output, attn_weight
