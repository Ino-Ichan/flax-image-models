import jax
from .registory import model_entorypoint
from .helpers import load_params

__all__ = ['create_model']


def create_model(
    rng: jax.random.PRNGKey,
    model_name: str,
    pretrained: bool = False,
    checkpoint_path: str = '',
    **kwargs,
):
    """Create a model
    Lookup model's entrypoint function and pass relevant args to create a new model.
    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        checkpoint_path (str): path of checkpoint to load _after_ the model is initialized
    """

    create_fn = model_entorypoint(model_name=model_name)
    rng, model, params = create_fn(rng, pretrained=pretrained, **kwargs)

    if checkpoint_path:
        params = load_params(params=params, path=checkpoint_path)

    return rng, model, params
