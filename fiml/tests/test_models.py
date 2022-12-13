import os
import numpy as np
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn

from fiml.models import ConvNeXt

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = 'platform'


@jax.jit
def test_convnext():
    rng = jax.random.PRNGKey(0)
    rng, rng_dp = jax.random.split(rng, 2)

    shape = (4, 224, 224, 3)
    inp = np.random.randn(*shape)

    model_share = partial(
        ConvNeXt,
        num_classes=1000,
        in_chs=3,
        global_pool='avg',
        output_stride=32,
        kernel_size=7,
        ls_init_value=1e-6,
        patch_size=4,
        head_init_scale=1.,
        head_norm_first=False,
        conv_bias=True,
        act_layer=nn.gelu,
        drop_rate=0.5,
        drop_path_rate=0.5,
        dtype=jnp.float32,
    )

    configs = [
        # [depths, dims]
        [(3, 3, 9, 3), (96, 192, 384, 768)],  # tiny
        [(3, 3, 27, 3), (96, 192, 384, 768)],  # small
        [(3, 3, 27, 3), (128, 256, 512, 1204)],  # base
        [(3, 3, 27, 3), (256, 512, 1024, 2048)],  # base
    ]
    for depths, dims in configs:
        conv_next = model_share(
            depths=depths,
            dims=dims,
        )
        params = conv_next.init(
            {
                "params": rng,
                "drop_path": rng,
                "dropout": rng,
            },
            jnp.ones(shape=shape),
            deterministic=False)["params"]
        out = conv_next.apply({"params": params},
                              inp,
                              False,
                              rngs=({
                                  "drop_path": rng_dp,
                                  "dropout": rng_dp,
                              }))
        assert out.shape == (4, 1000)
