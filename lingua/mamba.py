import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Tuple
from .ssm import Mamba, InitArgs, MambaArgs
from lingua.transformer import InitStdFactor
from .norm import RMSNorm

@dataclass
class BaseMambaArgs:

    dim: int = 512
    n_layers: int = 8

    mamba : MambaArgs = field(default_factory=MambaArgs)

    vocab_size: int = -1


    """
    Enforces that the SwiGLU hidden layer size is a multiple
    of large power of 2.
    """

    norm_eps: float = 1e-5

    init_use_depth: bool = False
    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"

    init_args: InitArgs = field(default_factory=InitArgs)
    seed: int = 42



class MambaBlock(nn.Module):
    def __init__(self, args: BaseMambaArgs):
        super().__init__()

        self.mamba_norm = RMSNorm(args.dim, args.norm_eps)
        self.mamba = Mamba(
            args=args.mamba,
            dim=args.dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        tok_idx: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        ssm_impl: str = "ssm",
    ) -> torch.Tensor:
        x = x + self.mamba(
            self.mamba_norm(x), tok_idx=tok_idx, cu_seqlens=cu_seqlens, ssm_impl=ssm_impl
        )
        return x

    def init_weights(self, init_std=None, factor=1.0, init_args: InitArgs = InitArgs()):
        self.mamba_norm.reset_parameters()
        self.mamba.reset_parameters(init_std, factor, init_args)


class BaseMamba(nn.Module):
    def __init__(self, args: BaseMambaArgs):
        super().__init__()
        self.model_dim = args.dim
        self.init_base_std = args.init_base_std

        self.init_args = args.init_args
        self.init_std_factor = InitStdFactor(args.init_std_factor)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(MambaBlock(args))

    def forward(
        self,
        h: torch.Tensor,
        tok_idx: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        ssm_impl: str = "ssm",
    ) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, tok_idx=tok_idx, cu_seqlens=cu_seqlens, ssm_impl=ssm_impl)
        return h

    def reset_parameters(self):
        pass

    def init_weights(self):
        self.reset_parameters()
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.model_dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(self.init_base_std, factor)
