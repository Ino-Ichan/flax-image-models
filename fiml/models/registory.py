"""Model registory
"""
import re
import fnmatch
from typing import Union, List

__all__ = ['registor_model', 'list_models', 'model_entorypoint']

_model_entrypoints = {}


def registor_model(fn):
    model_name = fn.__name__
    _model_entrypoints[model_name] = fn


def _natural_key(string_):
    return [
        int(s) if s.isdigit() else s
        for s in re.split(r'(\d+)', string_.lower())
    ]


def list_models(filter: Union[str, List[str]] = ''):
    """ Return list of available model names, sorted alphabetically
    Args:
        filter (str) - Wildcard filter string that works with fnmatch
    """
    all_models = _model_entrypoints.keys()

    if filter:
        models = []
        include_filters = filter if isinstance(filter,
                                               (tuple, list)) else [filter]
        for f in include_filters:
            include_models = fnmatch.filter(all_models, include_filters)
            if len(include_filters):
                models = set(models).union(include_models)
    else:
        models = all_models

    return list(sorted(models, key=_natural_key))


def model_entorypoint(model_name):
    return _model_entrypoints[model_name]
