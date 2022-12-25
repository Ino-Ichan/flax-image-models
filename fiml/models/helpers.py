import os
import sys
import urllib
from appdirs import user_cache_dir

from flax.serialization import to_bytes, from_bytes
from flax.linen import FrozenDict

__all__ = ['save_params', 'load_params', 'download_params']


def save_params(params: FrozenDict, path: str) -> None:
    serialized_params = to_bytes(params)
    with open(path, 'wb') as f:
        f.write(serialized_params)


def load_params(params: FrozenDict, path: str) -> FrozenDict:
    with open(path, 'rb') as f:
        serialized_params = f.read()

    return FrozenDict(from_bytes(params, serialized_params))


def progress(block_count: int, block_size: int, total_size: int):
    # urllib.request.urlretrieve report hook
    percentage = 100.0 * block_count * block_size / total_size
    sys.stdout.write(
        f"Download weight: {percentage:.2f}% ( {total_size / 1024**2:.1f} MB )\r"
    )


def download_params(url: str):
    weight_dir = user_cache_dir("fiml")
    if os.path.exists(weight_dir) == False:
        os.makedirs(weight_dir)

    model_name = url.split("/")[-1]
    file_path = os.path.join(weight_dir, model_name)
    if os.path.exists(file_path):
        print(f'{model_name} is already downloaded at {file_path}.')
    else:
        urllib.request.urlretrieve(url, file_path, progress)
        print(f'{model_name} is successfully downloaded at {file_path}.')
    return file_path
