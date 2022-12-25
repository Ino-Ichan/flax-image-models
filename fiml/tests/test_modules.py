import os
import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn

from fiml.layers import (Dropout, DropPath, Mlp, ConvMlp)

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = 'platform'


def test_dropout():
    rng = jax.random.PRNGKey(0)
    rng, dropout_rng0 = jax.random.split(rng, 2)

    shape = (4, 64, 64, 3)
    inp = np.random.randn(*shape)
    dropout = Dropout(drop=0.5, deterministic=False)
    # # params is nn.FrozenDict({}) since Dropout has no params
    # params = dropout.init({"params": params, "dropout": dropout_rng0}, jnp.ones(shape=shape))
    out = dropout.apply({"params": nn.FrozenDict({})},
                        inp,
                        rngs={"dropout": dropout_rng0})
    assert out.shape == inp.shape


def test_droppath():
    rng = jax.random.PRNGKey(0)
    rng, droppath_rng0 = jax.random.split(rng, 2)

    shape = (4, 64, 4, 4)
    inp = np.random.randn(*shape)
    droppath = DropPath(drop=0.5, deterministic=False)
    out = droppath.apply({"params": nn.FrozenDict({})},
                         inp,
                         rngs=({
                             "drop_path": droppath_rng0
                         }))
    assert out.shape == inp.shape


@jax.jit
def test_mlp():
    rng = jax.random.PRNGKey(0)

    in_shape = (8, 4, 4, 16)
    out_shape = (8, 4, 4, 16)

    inp = np.random.randn(*in_shape)

    mlp = Mlp(hidden_features=32, out_features=16)
    params = mlp.init({"params": rng}, jnp.ones(in_shape))["params"]
    # # if you didin't set "dropout", then automatically set dropout rng from mlp "params" rng
    # out = mlp.apply({"params": params}, inp, rngs={"dropout": rng})
    out = mlp.apply({"params": params}, inp)
    assert out.shape == out_shape


@jax.jit
def test_convmlp():
    rng = jax.random.PRNGKey(0)

    in_shape = (8, 4, 4, 16)
    out_shape = (8, 4, 4, 16)

    inp = np.random.randn(*in_shape)

    mlp = ConvMlp(hidden_features=32, out_features=16)
    params = mlp.init({"params": rng}, jnp.ones(in_shape))["params"]
    # # if you didin't set "dropout", then automatically set dropout rng from mlp "params" rng
    # out = mlp.apply({"params": params}, inp, rngs={"dropout": rng})
    out = mlp.apply({"params": params}, inp)
    assert out.shape == out_shape
