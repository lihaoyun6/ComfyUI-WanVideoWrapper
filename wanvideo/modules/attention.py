# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from .draft_attention import Draft_Attention

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    @torch.compiler.disable()
    def sageattn_func(q, k, v, attn_mask=None, dropout_p=0, is_causal=False):
        if q.dtype == torch.float32:
            return sageattn(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal).to(torch.float32)
        else:
            return sageattn(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
except Exception as e:
    print(f"Warning: Could not load sageattention: {str(e)}")
    if isinstance(e, ModuleNotFoundError):
        print("sageattention package is not installed")
    elif isinstance(e, ImportError) and "DLL" in str(e):
        print("sageattention DLL loading error")
    sageattn_func = None
import warnings

__all__ = [
    'flash_attention',
    'attention',
]

def draft_attention_with_fallback(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    sparsity_ratio=0.75,
    version=None,
    idx_block=None,
    current_timestep=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256
    
    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
    
    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)
    
    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))
        
    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))
        
    q = q.to(v.dtype)
    k = k.to(v.dtype)
    
    if q_scale is not None:
        q = q * q_scale
        
    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )
        
    # =====================================================================
    # # Dense:    # todo
    # x = flash_attn.flash_attn_varlen_func(
    #     q=q,
    #     k=k,
    #     v=v,
    #     cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
    #         0, dtype=torch.int32).to(q.device, non_blocking=True),
    #     cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
    #         0, dtype=torch.int32).to(q.device, non_blocking=True),
    #     max_seqlen_q=lq,
    #     max_seqlen_k=lk,
    #     dropout_p=dropout_p,
    #     softmax_scale=softmax_scale,
    #     causal=causal,
    #     window_size=window_size,
    #     deterministic=deterministic).unflatten(0, (b, lq))
        
    # # output
    # return x.type(out_dtype)
    # =====================================================================
        
    # self attention: q, k, v shape: [32256, 40, 128];      [80640, 40, 128] 80640=21x 48x80
    # cross attention: q [32256, 40, 128], k, v [512, 40, 128]
        
    assert q.shape[0] == 32_256 or 80_640, "Currently, we only support 768*512 and 1280*768 resolution, and we will implement the padding for any-resolution generation in the future."
    
    # =====================================================================
    # Xuan: sparse
    if q.shape[0] != k.shape[0] or (idx_block is not None and idx_block < 1) or (current_timestep > 925):
        # trick from https://github.com/svg-project/Sparse-VideoGen/blob/079364dc2e4ca6cd0c26c8c45eafeb9fbf51ef8e/svg/models/wan/attention.py#L212
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))
        # output
        return x.type(out_dtype)
    
    draft_attention = Draft_Attention(
            pool_h=8,
            pool_w=16,
            latent_h=32 if q.shape[0] == 32256 else 48,
            latent_w=48 if q.shape[0] == 32256 else 80,
            visual_len=q.shape[0],
            text_len=0,
            sparsity_ratio=sparsity_ratio
        )
    x = draft_attention(
        q,
        k,
        v,
        attn_mask=None,
        causal=causal,
        drop_rate=dropout_p,
        cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
            0, dtype=torch.int32).to(q.device, non_blocking=True),
        cu_seqlens_kv=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
            0, dtype=torch.int32).to(q.device, non_blocking=True),
        max_seqlen_q=lq,
        max_seqlen_kv=lk,
        batch_size=1,
    )
    
    # output
    return x.type(out_dtype)
    # =====================================================================


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    #assert dtype in half_dtypes
    #assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    attention_mode='sdpa',
    fa_version=None,
    idx_block=None,
    current_timestep=None,
):  
    if "flash" in attention_mode:
        if attention_mode == 'flash_attn_2':
            fa_version = 2
        elif attention_mode == 'flash_attn_3':
            fa_version = 3
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    elif attention_mode == 'sdpa':
        return torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).contiguous()
    elif attention_mode == 'sageattn':
        return sageattn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).contiguous()
    elif "draft_attn" in attention_mode:
        assert FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE, "We uses flash attention as the fallback solution, please install flash-attn first!"
        if "fa_2" in attention_mode:
            fa_version = 2
        elif "fa_3" in attention_mode:
            fa_version = 3
        sparsity_ratio = float(attention_mode.split("@")[1]) / 100
        return draft_attention_with_fallback(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            sparsity_ratio=sparsity_ratio,
            version=fa_version,
            idx_block=idx_block,
            current_timestep=idx_block,
        )