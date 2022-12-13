from functools import partial
from typing import Any, Callable, Sequence, Tuple
import sys

import jax
import jax.numpy as jnp
import flax.linen as nn

from .drop import Dropout

ModuleDef = Any
Dtype = Any


class Mlp(nn.Module):
    hidden_features: int
    out_features: int = None
    act_layer: Callable = nn.gelu
    bias: bool = True
    drop: float = 0.
    dtype: Dtype = jnp.float32

    deterministic: bool = None  # for dropout

    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        hidden_features = self.hidden_features
        out_features = self.out_features
        deterministic = nn.merge_param('deterministic', self.deterministic,
                                       deterministic)

        x = nn.Dense(hidden_features, use_bias=self.bias, dtype=self.dtype)(x)
        x = self.act_layer(x)
        x = Dropout(self.drop, deterministic=deterministic)(x)
        x = nn.Dense(out_features, use_bias=self.bias, dtype=self.dtype)(x)
        x = Dropout(self.drop, deterministic=deterministic)(x)
        return x


class ConvMlp(nn.Module):
    hidden_features: int
    out_features: int = None
    deterministic: bool = None  # for dropout
    act_layer: Callable = nn.gelu
    bias: bool = True
    drop: float = 0.
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        hidden_features = self.hidden_features
        out_features = self.out_features
        deterministic = nn.merge_param('deterministic', self.deterministic,
                                       deterministic)

        conv_pw = partial(nn.Conv, kernel_size=[1, 1], dtype=self.dtype)

        x = conv_pw(hidden_features)(x)
        x = self.act_layer(x)
        x = Dropout(self.drop, deterministic=deterministic)(x)
        x = conv_pw(out_features)(x)
        x = Dropout(self.drop, deterministic=deterministic)(x)
        return x
