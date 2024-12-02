from functools import partial
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
    _mask_mod_signature,
    and_masks,
    create_block_mask
)
from xformers.ops import fmha, AttentionBias
from torch.nn import functional as F
import math
from lingua.norm.rms_norm import RMSNorm
from flash_attn import flash_attn_func

flex_attention_comp = torch.compile(flex_attention)

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def sliding_window_mask(b, h, q_idx, kv_idx, slinding_window):
    return (q_idx - kv_idx) <= slinding_window


def create_causal_mask(seqlen, attn_impl, sliding_window):
    if sliding_window is not None and attn_impl == "xformers":
        return fmha.attn_bias.LocalAttentionFromBottomRightMask(
            window_left=sliding_window - 1, window_right=0
        )
    elif attn_impl == "xformers":
        return fmha.attn_bias.LowerTriangularMask()
    elif attn_impl == "sdpa":
        return "causal"
    elif attn_impl == "flex_attention":
        mask_func = causal_mask
        if sliding_window is not None:
            sw_mask = partial(
                sliding_window_mask, sliding_window=sliding_window
            )
            mask_func = and_masks(mask_func, sw_mask)
        #doc_mask = generate_doc_mask_mod(mask_func, lengths, kv_lengths)
        #mask_func = and_masks(mask_func, doc_mask) 
        return create_block_mask(mask_func, None, None, seqlen, seqlen)
    else:
        raise NotImplementedError(
            f"Attention {attn_impl} with {sliding_window} sliding window not implemented"
        )


def lengths_to_start_ids(lengths):
    doc_start = lengths.cumsum(0)
    doc_start = doc_start.roll(1)
    doc_start[0] = 0
    return doc_start


def lengths_to_local_ids(lengths):
    assert lengths.ndim == 1
    nb_seqs = lengths.size(0)
    total_seqlen = lengths.sum()
    # This gives the document id of each token
    doc_id = torch.repeat_interleave(lengths)
    # Compute document start for each document
    doc_start = lengths_to_start_ids(lengths)
    # Compute document start for each token
    doc_start = doc_start[doc_id]
    # Compute the position of each token within each document
    tok_id = torch.arange(total_seqlen, device=lengths.device) - doc_start

    return doc_id, tok_id

def generate_doc_mask_mod(
    mask_mod: _mask_mod_signature,
    lengths: torch.Tensor,
    kv_lengths: Optional[torch.Tensor] = None,
) -> _mask_mod_signature:
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        mask_mod: The mask mod to apply to the documents
        lengths: Lengths of each document

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.

    Example:

    - Square mask
      doc_mask         lengths
      a a b b b c c    2 3 2
    a 1 0 0 0 0 0 0
    a 1 1 0 0 0 0 0
    b 0 0 1 0 0 0 0
    b 0 0 1 1 0 0 0
    b 0 0 1 1 1 0 0
    c 0 0 0 0 0 1 0
    c 0 0 0 0 0 1 1

    """
    kv_lengths = kv_lengths if kv_lengths is not None else lengths
    q_document_id, q_token_id = lengths_to_local_ids(lengths)
    kv_document_id, kv_token_id = lengths_to_local_ids(kv_lengths)
    q_max_idx = lengths.sum() - 1
    kv_max_idx = kv_lengths.sum() - 1

    def doc_mask_mod(b, h, q_idx, kv_idx):
        q_idx_cap = torch.minimum(q_max_idx, q_idx)
        kv_idx_cap = torch.minimum(kv_max_idx, kv_idx)
        valid_idx = (q_idx <= q_max_idx) & (kv_idx <= kv_max_idx)
        same_doc = q_document_id[q_idx_cap] == kv_document_id[kv_idx_cap]
        q_logical = q_token_id[q_idx_cap]
        kv_logical = kv_token_id[kv_idx_cap]
        inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return same_doc & inner_mask & valid_idx

    return doc_mask_mod

def repeat_kv(x: torch.Tensor, n_rep: int, dim: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    assert dim == 2, "Only dim=2 is supported. Check the implementation for other dims."
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor, seq_dim: int):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        seq_dim (int): Sequence dimension index.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= seq_dim < ndim
    assert freqs_cis.shape == (
        x.shape[seq_dim],
        x.shape[-3],
        2,
        2,
    ), f"freqs_cis vs x: {(freqs_cis.shape, x.shape)}"
    shape = [
        d if i == seq_dim or i == ndim - 3 else 1 for i, d in enumerate(x.shape[:-2])
    ] + [2, 2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    seq_dim: int,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    print("xq : ", xq.shape)
    print("xk : ", xk.shape)
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    freqs_cis = reshape_for_broadcast(
        freqs_cis, xq_, seq_dim
    ).float()  # S D/2 2 2 -> 1 S 1 D/2 2 2
    xq_out = (xq_ * freqs_cis).sum(5).flatten(3)
    xk_out = (xk_ * freqs_cis).sum(5).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx



def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class DiffAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        block_id: int = 0,
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim // 2
        self.rope_theta = rope_theta

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads


        self.wq = nn.Linear(
            dim,
            n_heads * (self.head_dim * 2),
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            n_kv_heads * (self.head_dim * 2),
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            n_kv_heads * (self.head_dim * 2),
            bias=False,
        )

        self.wo = nn.Linear(
            n_heads * (self.head_dim * 2),
            dim,
            bias=False,
        )
        self.lambda_init = lambda_init_fn(block_id)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "flex_attention",
    ) -> torch.Tensor:
        # B S D
        bsz, seq_len, dim = x.shape
        xq = self.wq(x.view_as(x))
        xk = self.wk(x.view_as(x))
        xv = self.wv(x.view_as(x))

        output_shape = xq.shape

        # B S D -> B S H D
        xq = xq.view(bsz, seq_len, self.n_heads, 2*self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, 2*self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, 2*self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])

        # This condition helps us be easily compatible
        # with inference by adding a pluggable KVCache
        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)

        xq = xq.reshape(bsz, seq_len, self.n_heads, 2, self.head_dim)
        xk = xk.reshape(bsz, seq_len, self.n_kv_heads, 2, self.head_dim)
        
        xq1, xq2 = xq[:, :, :, 0], xq[:, :, :, 1]
        xk1, xk2 = xk[:, :, :, 0], xk[:, :, :, 1]

        if attn_impl != "flex_attention":
            raise NotImplementedError(
                f"Diff Attention implementation on {attn_impl} not supported"
            )
                        
        assert mask is None or isinstance(mask, BlockMask)
        #xq1 = xq1.reshape(bsz, seq_len, self.n_heads, self.head_dim)
        #xk1 = xk1.reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)
        #xq1, xk1, _xv = map(lambda e: e.transpose(1, 2), (xq1, xk1, xv))
        #print("xq1 : ", xq1.shape)
        #print("xk1 : ", xk1.shape)
        #print("_xv : ", _xv.shape)
        attn1 = flash_attn_func(xq1, xk1, xv, causal=True)
        attn1 = attn1.transpose(1, 2).contiguous()  # B H S D -> B S H D

        #xq2 = xq2.reshape(bsz, seq_len, self.n_heads, self.head_dim)
        #xk2 = xk2.reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)
        #print("xq2 : ", xq2.shape)
        #print("xk2 : ", xk2.shape)
        #print("_xv : ", _xv.shape)
        #xq2, xk2, _xv = map(lambda e: e.transpose(1, 2), (xq2, xk2, xv))
        attn2 = flash_attn_func(xq2, xk2, xv, causal=True)
        attn2 = attn2.transpose(1, 2).contiguous()
        
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(xq)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(xq)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2
        
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(bsz, seq_len, self.n_heads * 2 * self.head_dim)

        attn = self.wo(attn.reshape(output_shape))

        return attn

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))
        init_std = init_std / factor

        for w in [self.wq, self.wk, self.wv, self.wo]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )
