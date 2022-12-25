from functools import partial
from typing import Any, Tuple, Optional, List
from dataclasses import field
import sys

import numpy as np

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

from ..layers import Mlp, DropPath
from .registory import registor_model
from .helpers import load_params, download_params

ModuleDef = Any
Dtype = Any


class ConvNeXtBlock(nn.Module):
    in_chs: int
    out_chs: int = None
    kernel_size: int = 7
    stride: int = 1
    dilation: int = 1
    mlp_ratio: int = 4
    conv_bias: bool = True
    ls_init_value: float = 1e-6
    act_layer: ModuleDef = nn.gelu
    drop_path: float = 0.
    dtype: Dtype = jnp.float32

    deterministic: Optional[bool] = None

    def init_fn(self, key, shape, values):
        return jnp.ones(shape=shape) * values

    @nn.compact
    def __call__(self, x, deterministic: Optional[bool] = None):
        deterministic = nn.merge_param('deterministic', self.deterministic,
                                       deterministic)

        in_chs = self.in_chs
        out_chs = self.out_chs or in_chs

        conv_dw = partial(nn.Conv,
                          features=out_chs,
                          kernel_size=[self.kernel_size, self.kernel_size],
                          strides=self.stride,
                          kernel_dilation=self.dilation,
                          use_bias=self.conv_bias,
                          feature_group_count=in_chs,
                          dtype=self.dtype)
        layer_norm = partial(nn.LayerNorm,
                             use_bias=self.conv_bias,
                             dtype=self.dtype)
        mlp = partial(Mlp,
                      hidden_features=int(self.mlp_ratio * out_chs),
                      out_features=out_chs,
                      act_layer=self.act_layer,
                      dtype=self.dtype)

        shortcut = x
        x = conv_dw()(x)
        x = layer_norm()(x)
        x = mlp()(x)
        if self.ls_init_value is not None:
            gamma = self.param('gamma', self.init_fn, (x.shape[-1], ),
                               self.ls_init_value)
            x = x * gamma

        x = DropPath(self.drop_path)(x, deterministic=deterministic)
        return x + shortcut


class ConvNextStage(nn.Module):
    in_chs: int
    out_chs: int
    kernel_size: int = 7
    stride: int = 1
    depth: int = 2
    dilation: List[int] = field(default_factory=[1, 1])
    drop_path_rates: List[float] = field(default_factory=[0., 0.])
    ls_init_value: float = 1.
    conv_bias: bool = True
    act_layer: ModuleDef = nn.gelu
    dtype: Dtype = jnp.float32

    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, x, deterministic: Optional[bool] = None):
        deterministic = nn.merge_param('deterministic', self.deterministic,
                                       deterministic)
        in_chs = self.in_chs
        out_chs = self.out_chs

        if in_chs != out_chs or self.stride > 1 or self.dilation[
                0] != self.dilation[1]:
            ds_ks = 2 if self.stride > 1 or self.dilation[0] != self.dilation[
                1] else 1
            pad = 'SAME' if self.dilation[1] > 1 else 0

            layer_norm = nn.LayerNorm(use_bias=self.conv_bias,
                                      dtype=self.dtype)

            conv_ds = nn.Conv(features=self.out_chs,
                              kernel_size=[ds_ks, ds_ks],
                              strides=self.stride,
                              kernel_dilation=self.dilation[0],
                              use_bias=self.conv_bias,
                              padding=pad,
                              dtype=self.dtype)

            x = nn.Sequential([layer_norm, conv_ds])(x)
            in_chs = out_chs

        drop_path_rates = self.drop_path_rates or [0.] * self.depth

        for i in range(self.depth):
            x = ConvNeXtBlock(in_chs=in_chs,
                              out_chs=out_chs,
                              kernel_size=self.kernel_size,
                              dilation=self.dilation[1],
                              drop_path=drop_path_rates[i],
                              ls_init_value=self.ls_init_value,
                              conv_bias=self.conv_bias,
                              act_layer=self.act_layer,
                              dtype=self.dtype)(x, deterministic=deterministic)
            in_chs = out_chs

        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A ConvNeXt impl of : `A ConvNet for the 2020s`  - https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chs (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        global_pool (str): Type of global pooling
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (tuple(int)): Feature dimension at each stage. Default: [96, 192, 384, 768]
        kernel_size (int): Kernel size of depth-wise conv. Default: 7
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
        patch_size (int): Patch size of stem. Default: 4
        act_layer (ModuleDef): Activation layer. Default: nn.gelu
        drop_rate (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        dtype (Dtype): Dtype of each layer. Default: jnp.float32
        deterministic (Optional (bool)): If True, no dropout/droppath mask is sampled. Default: None
    """

    in_chs: int = 3
    num_classes: int = 1000
    global_pool: str = 'avg'
    output_stride: int = 32
    depths: Tuple[int] = field(default_factory=(3, 3, 9, 3))
    dims: Tuple[int] = field(default_factory=(96, 192, 384, 768))
    kernel_size: int = 7
    ls_init_value: float = 1e-6
    patch_size: int = 4
    conv_bias: bool = True
    act_layer: ModuleDef = nn.gelu
    drop_rate: float = 0.
    drop_path_rate: float = 0.
    dtype: Dtype = jnp.float32
    deterministic: Optional[bool] = None

    def setup(self) -> None:
        assert self.output_stride in (8, 16, 32)
        kernel_sizes = (self.kernel_size, ) * 4

        self.stem = nn.Sequential([
            nn.Conv(self.dims[0],
                    kernel_size=[self.patch_size, self.patch_size],
                    strides=self.patch_size,
                    use_bias=self.conv_bias),
            nn.LayerNorm()
        ])

        dp_rates = [
            x.tolist() for x in np.split(
                np.linspace(0, self.drop_path_rate, sum(self.depths)),
                np.cumsum(self.depths)) if len(x) > 0
        ]

        prev_chs = self.dims[0]
        curr_stride = self.patch_size
        dilation = 1

        stages = []
        for i in range(4):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            if curr_stride >= self.output_stride and stride > 1:
                dilation *= stride
                stride = 1
            curr_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2
            out_chs = self.dims[i]

            stages.append(
                ConvNextStage(in_chs=prev_chs,
                              out_chs=out_chs,
                              kernel_size=kernel_sizes[i],
                              stride=stride,
                              depth=self.depths[i],
                              dilation=(first_dilation, dilation),
                              drop_path_rates=dp_rates[i],
                              ls_init_value=self.ls_init_value,
                              conv_bias=self.conv_bias,
                              act_layer=self.act_layer,
                              dtype=self.dtype))
            prev_chs = out_chs
        self.stages = stages

        self.head_norm = nn.LayerNorm()
        self.head_dropout = nn.Dropout(self.drop_rate)
        if self.num_classes > 0:
            self.head_dense = nn.Dense(self.num_classes, dtype=self.dtype)

    def __call__(self, x, train: Optional[bool] = None):
        deterministic = train
        deterministic = nn.merge_param('deterministic', self.deterministic,
                                       deterministic)

        x = self.stem(x)
        for stage in self.stages:
            x = stage(x, deterministic=deterministic)
        x = self.global_pooling(x)
        x = self.head_norm(x)
        x = self.head_dropout(x, deterministic=deterministic)
        if self.num_classes > 0:
            x = self.head_dense(x)
        return x

    def global_pooling(self, x):
        if self.global_pool == 'avg':
            x = nn.avg_pool(x, (x.shape[1], x.shape[2]))
        else:
            x = nn.max_pool(x, (x.shape[1], x.shape[2]))
        return x.reshape(x.shape[0], -1)


_params_url_dict = {
    "convnext_tiny":
    "https://github.com/Ino-Ichan/flax-image-models/releases/download/v0.1-convnext/fbaipublicfiles-convnext_tiny.params",
    "convnext_small":
    "https://github.com/Ino-Ichan/flax-image-models/releases/download/v0.1-convnext/fbaipublicfiles-convnext_small.params",
    "convnext_base":
    "https://github.com/Ino-Ichan/flax-image-models/releases/download/v0.1-convnext/fbaipublicfiles-convnext_base.params",
    "convnext_large":
    "https://github.com/Ino-Ichan/flax-image-models/releases/download/v0.1-convnext/fbaipublicfiles-convnext_large.params",
}


def _create_convnext(rng: jax.random.PRNGKey,
                     model_name: str,
                     depths: Tuple[int],
                     dims: Tuple[int],
                     pretrained: bool = False,
                     num_classes: int = 1000,
                     input_shape: Tuple[int] = (224, 224, 3),
                     **kwargs):
    """
    kwargs: drop_rate, drop_path_rate, dtype
    """
    rng, rng_drop_path, rng_dropout = jax.random.split(rng, 3)

    conv_next = ConvNeXt(num_classes=num_classes,
                         in_chs=3,
                         global_pool='avg',
                         output_stride=32,
                         depths=depths,
                         dims=dims,
                         kernel_size=7,
                         ls_init_value=1e-6,
                         patch_size=4,
                         conv_bias=True,
                         act_layer=nn.gelu,
                         **kwargs)

    params = conv_next.init(
        {
            "params": rng,
            "drop_path": rng_drop_path,
            "dropout": rng_dropout,
        },
        jnp.ones(shape=(1, *input_shape)),
        deterministic=False)["params"]

    if pretrained:
        params_path = download_params(_params_url_dict[model_name])
        params_load = load_params(params=params, path=params_path)
        if num_classes != 1000:
            params_load = flax.core.unfreeze(params_load)
            params_load['head_norm'] = params['head_norm']
            if num_classes != 0:
                params_load['head_dense'] = params['head_dense']
            params_load = flax.core.freeze(params_load)
        params = params_load
    return conv_next, params


@registor_model
def convnext_tiny(rng: jax.random.PRNGKey,
                  pretrained: bool = False,
                  num_classes: int = 1000,
                  input_shape: Tuple[int] = (224, 224, 3),
                  **kwargs):
    conv_next, params = _create_convnext(
        rng=rng,
        model_name=sys._getframe().f_code.co_name,
        num_classes=num_classes,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        pretrained=pretrained,
        input_shape=input_shape,
        **kwargs)

    return rng, conv_next, params


@registor_model
def convnext_small(rng: jax.random.PRNGKey,
                   pretrained: bool = False,
                   num_classes: int = 1000,
                   input_shape: Tuple[int] = (224, 224, 3),
                   **kwargs):
    conv_next, params = _create_convnext(
        rng=rng,
        model_name=sys._getframe().f_code.co_name,
        num_classes=num_classes,
        depths=(3, 3, 27, 3),
        dims=(96, 192, 384, 768),
        pretrained=pretrained,
        input_shape=input_shape,
        **kwargs)

    return rng, conv_next, params


@registor_model
def convnext_base(rng: jax.random.PRNGKey,
                  pretrained: bool = False,
                  num_classes: int = 1000,
                  input_shape: Tuple[int] = (224, 224, 3),
                  **kwargs):
    conv_next, params = _create_convnext(
        rng=rng,
        model_name=sys._getframe().f_code.co_name,
        num_classes=num_classes,
        depths=(3, 3, 27, 3),
        dims=(128, 256, 512, 1024),
        pretrained=pretrained,
        input_shape=input_shape,
        **kwargs)

    return rng, conv_next, params


@registor_model
def convnext_large(rng: jax.random.PRNGKey,
                   pretrained: bool = False,
                   num_classes: int = 1000,
                   input_shape: Tuple[int] = (224, 224, 3),
                   **kwargs):
    conv_next, params = _create_convnext(
        rng=rng,
        model_name=sys._getframe().f_code.co_name,
        num_classes=num_classes,
        depths=(3, 3, 27, 3),
        dims=(192, 384, 768, 1536),
        pretrained=pretrained,
        input_shape=input_shape,
        **kwargs)

    return rng, conv_next, params
