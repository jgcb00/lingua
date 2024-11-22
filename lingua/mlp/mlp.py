import torch
from torch import nn
from .swish_mlp import SwishFeedForward
from .relu_mlp import ReluFeedForward
from .FAN import FanFeedForward
from .XNet import XnetFeedForward
from dataclasses import dataclass
from typing import Optional
@dataclass
class MlpArgs :
    ffn_type : str = "swish"
    multiple_of : int = 256
    ffn_dim_multiplier : Optional[float] = None


class MLP(nn.Module):
    def __init__(self, args : MlpArgs, dim : int, block_id : int = 0) -> None:
        super().__init__()
                
        if args.ffn_type == "swish":
            self.feed_forward = SwishFeedForward
        elif args.ffn_type == "relu":
            self.feed_forward = ReluFeedForward
        elif args.ffn_type == "fan":
            self.feed_forward = FanFeedForward
        elif args.ffn_type == "xnet":
            self.feed_forward = XnetFeedForward
        else:
            raise ValueError(f"Unknown ffn_type: {args.ffn_type}")
        
        self.feed_forward = self.feed_forward(
                dim=dim,
                hidden_dim=4 * dim,
                multiple_of=args.multiple_of,
                ffn_dim_multiplier=args.ffn_dim_multiplier,
            )
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(x)
    
    def reset_parameters(self, init_std=None, factor=1.0):
        self.feed_forward.reset_parameters(init_std, factor)