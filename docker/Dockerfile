FROM nvcr.io/nvidia/tensorflow:22.08-tf2-py3

WORKDIR /workspace

RUN pip install --upgrade pip

# Installs the wheel compatible with Cuda >= 11.8 and cudnn >= 8.6
RUN pip install --upgrade pip && pip install "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install flax==0.6.3

# tmux
RUN apt-get update && apt-get install -y tmux

# opencv
RUN pip install opencv-python==4.5.5.62
RUN apt-get update && apt-get install -y libgl1-mesa-dev

WORKDIR /codes