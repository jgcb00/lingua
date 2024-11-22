import torch.nn as nn
import torch
from typing import Optional, Union
from .softmax_attention import SoftmaxAttention
from .diff_attention import DiffAttention
from torch.nn.attention.flex_attention import BlockMask
from xformers.ops import AttentionBias
from dataclasses import dataclass

@dataclass
class AttentionArgs:
    n_heads: int = 16
    head_dim: Optional[int] = None
    n_kv_heads: Optional[int] = None
    rope_theta: float = 10000.0
    type: str = "base"
    
    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_heads % self.n_kv_heads == 0



class Attention(nn.Module):
    def __init__(self, args : AttentionArgs, dim : int, block_id=0) -> None:
        super().__init__()
        assert (args.head_dim is not None) or (
            args.n_heads is not None
        ), "Should specify at least head_dim or n_heads"
        self.head_dim = args.head_dim or dim // args.n_heads
        self.n_heads = args.n_heads or dim // self.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads
        
        assert args.dim % args.n_heads == 0
        
        if args.type == "base":
            self.attention = SoftmaxAttention
        elif args.type == "diff_attention":
            self.attention = DiffAttention
        else:
            raise ValueError(f"Unknown type: {args.type}")
            
        self.attention = self.attention(
            dim=dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
            block_id=block_id,
        )
        
    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "flex_attention",
    ) -> torch.Tensor:
        return self.attention(
            x,
            freq_cis,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl=attn_impl,
        )
        
    def reset_parameters(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)