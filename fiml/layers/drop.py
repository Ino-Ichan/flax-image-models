from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional
import sys

import jax
import jax.numpy as jnp
import flax.linen as nn

ModuleDef = Any


class Dropout(nn.Module):
    """ Dropout layer.
        Mainly copied from rwightman's efficientnet-jax codes
        Attributes:
            drop: the dropout probability.  (_not_ the keep rate!)
            deterministic: for inference
    """
    drop: float
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self,
                 x,
                 rng: jax.random.PRNGKey = None,
                 deterministic: Optional[bool] = None):
        deterministic = nn.merge_param('deterministic', self.deterministic,
                                       deterministic)
        if deterministic or self.drop == 0:
            return x
        keep_prob = 1. - self.drop
        if rng == None:
            rng = self.make_rng('dropout')
        mask = jax.random.bernoulli(rng, p=keep_prob, shape=x.shape)
        return jax.lax.select(mask, x / keep_prob, jnp.zeros_like(x))


class DropPath(nn.Module):
    """ Dropout layer.
        Mainly copied from rwightman's efficientnet-jax codes
        Attributes:
            drop: the dropout probability.  (_not_ the keep rate!)
            deterministic: for inference
    """
    drop: float = 0.
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self,
                 x,
                 rng: jax.random.PRNGKey = None,
                 deterministic: Optional[bool] = None):
        deterministic = nn.merge_param('deterministic', self.deterministic,
                                       deterministic)
        if deterministic or self.drop == 0:
            return x
        keep_prob = 1. - self.drop
        if rng == None:
            rng = self.make_rng('drop_path')
        shape = (x.shape[0], ) + (1, ) * (len(x.shape) - 1)
        mask = jax.random.bernoulli(rng, p=keep_prob, shape=shape)
        mask = jnp.broadcast_to(mask, x.shape)
        return jax.lax.select(mask, on_true=x, on_false=jnp.zeros_like(x))
