FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    vim \
    wget \
    ca-certificates \
    libsndfile1-dev \
    libgl1 \
    python3 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

## copy the repo
COPY ./ /workspace/p2-diffusion/
WORKDIR /workspace/p2-diffusion/

# install deps
RUN conda init bash \
    && . ~/.bashrc \
    && conda create -y --name p2 \
    && conda activate p2 \
    && pip install . \
    && conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia \
    && conda install -y -c conda-forge mpi4py mpich \
    && pip install numpy mpi4py blobfile