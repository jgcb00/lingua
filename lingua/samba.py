from dataclasses import dataclass, field
from .mlp import MLP, MlpArgs
from .norm import RMSNorm
from .ssm import Mamba, MambaArgs, InitArgs
from .attention import Attention, BlockMask, AttentionBias, AttentionArgs
import torch
import torch.nn as nn
from typing import Optional, Union
from lingua.transformer import InitStdFactor, RotaryEmbedding

@dataclass
class BaseSambaArgs:

    mamba: MambaArgs = field(default_factory=MambaArgs)
    attention: AttentionArgs = field(default_factory=AttentionArgs)
    mlp : MlpArgs = field(default_factory=MlpArgs)
    
    dim: int = 1024
    n_layers: int = 6
        
    norm_eps: float = 1e-5
    max_seqlen: int = 4096
    
    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"
    init_use_depth: bool = False

    init_args: InitArgs = field(default_factory=InitArgs)
    seed: int = 42
    

class SambaBlock(nn.Module):
    def __init__(self, args: BaseSambaArgs, block_id: int = 0):
        super().__init__()
        self.mamba_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.mamba = Mamba(args.mamba, dim=args.dim, block_id=block_id)
        self.mlp_1_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        self.mlp_1 = MLP(args.mlp, args.dim, block_id)
        self.attention_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        self.attention = Attention(args.attention, block_id)
        self.mlp_2_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        self.mlp_2 = MLP(args.mlp, args.dim, block_id)
        self.block_id = block_id
        self.args = args
    
    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "flex_attention",
        ssm_impl: str = "ssm",

    ) -> torch.Tensor:
        h = x + self.mamba(self.mamba_norm(x), tok_idx=tok_idx, cu_seqlens=cu_seqlens, ssm_impl=ssm_impl)
        h = h + self.mlp_1(self.mlp_1_norm(h))
        h = h + self.attention(self.attention_norm(h), freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
        out = h + self.mlp_2(self.mlp_2_norm(h))
        return out
    
    def init_weights(self, init_std=None, factor=1.0):     
        self.mamba.reset_parameters(init_std, factor)
        self.mamba_norm.reset_parameters()
        self.mlp_1.reset_parameters(init_std, factor)
        self.mlp_1_norm.reset_parameters()
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()
        self.mlp_2.reset_parameters(init_std, factor)
        self.mlp_2_norm.reset_parameters()
        
        
class BaseSamba(nn.Module):
    def __init__(self, args: BaseSambaArgs):
        super().__init__()
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_args = args.init_args
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.max_seqlen = args.max_seqlen
        self.rope_embeddings = RotaryEmbedding(
            theta=args.attention.rope_theta,
            head_dim=args.attention.head_dim or args.dim // args.attention.n_heads,
            max_seqlen=args.max_seqlen,
        )

        self.layers = nn.ModuleList()
        for block_id in range(args.n_layers):
            self.layers.append(SambaBlock(args, block_id))

    def forward(
        self,
        h,
        tok_idx: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "flex_attention",
        samba_impl: str = "samba",
    ):

        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)
        for i, layer in enumerate(self.layers):
            h = layer(
                h,
                freq_cis,
                tok_idx=tok_idx,
                cu_seqlens=cu_seqlens,
                mask=mask,
                attn_impl=attn_impl,
                samba_impl=samba_impl,
            )
        return h

    def reset_parameters(self):
        # Either use fixed base std or sqrt model dim
        self.rope_embeddings.reset_parameters()

    def init_weights(self):
        self.reset_parameters()
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(self.init_base_std, factor)
