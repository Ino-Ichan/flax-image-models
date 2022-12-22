# Flax Image Models

# Introduction

Flax Image Models (`fiml`) is a collection of image models with pretrained weights.

# Usage

## Installation
```
pip install git+https://github.com/Ino-Ichan/flax-image-models
```

## Docker
You can pull docker container for fiml. See [here](https://hub.docker.com/repository/docker/inoichan/fiml).
```
docker pull inoichan/fiml:latest
```

## Create models

```python
import jax
import fiml

rng = jax.random.PRNGKey(0)
rng, model, params = fiml.create_model(rng,
                                       'convnext_tiny',
                                       pretrained=True,
                                       num_classes=100)

inp = np.random.rand(4, 256, 256, 3)
out = model.apply(
    {"params": params},
    inp,
    deterministic=True,
)

print(out.shape)
# (4, 100)
```

# Models

## Architectures
- ConvNeXt [[paper](https://arxiv.org/abs/2201.03545)] [[github](https://github.com/facebookresearch/ConvNeXt)]

# Acknowledgement
- [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
- [google-research/vision_transformer](https://github.com/google-research/vision_transformer)

# License

This project is released under the [Apach 2.0 License](https://github.com/Ino-Ichan/flax-image-models/blob/main/LICENSE)