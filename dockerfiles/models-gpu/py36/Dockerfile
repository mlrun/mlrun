ARG CUDA_VER=10.1

FROM nvidia/cuda:${CUDA_VER}-cudnn7-devel-ubuntu18.04

# need to be redeclared since used in the from
ARG CUDA_VER

ENV PIP_NO_CACHE_DIR=1

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && \
    apt-mark hold libcublas-dev libcublas10 && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        git-core \
        graphviz \
        wget && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/installconda.sh && \
    /bin/bash ~/installconda.sh -b -p /opt/conda && \
    rm ~/installconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN conda config --add channels conda-forge && \
    conda update -n base -c defaults conda && \
    conda install -n base \
        python=3.6 \
        pip~=21.2.0 \
    && conda clean -aqy

RUN conda install -n base cmake cython \
    && conda clean -aqy

RUN conda install -n base -c rapidsai -c nvidia -c conda-forge "rapids==0.12" python=3.6 cudatoolkit=${CUDA_VER} \
    && conda clean -aqy

RUN conda install -n base pytorch::pytorch==1.7.0 pytorch::torchvision==0.8.0 \
    && conda clean -aqy

RUN python -m pip install 'tensorflow~=1.0' 'tensorflow-gpu~=1.0'

ARG OMPI=4.1.0

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    curl -o openmpi-${OMPI}.tar.gz https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OMPI}.tar.gz && \
    tar zxf openmpi-${OMPI}.tar.gz && \
    cd openmpi-${OMPI} && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j`nproc` all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi    

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# TODO: MAKEFLAGS="-j1" work around some transient concurrency problem with installing horovod remove it when
    #   possible (should be safe to remove if it works ~5 times without it)
RUN ldconfig /usr/local/cuda-10.1/targets/x86_64-linux/lib/stubs && \
    MAKEFLAGS="-j1" HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 \
    python -m pip install 'horovod~=0.20.0' && \
    ldconfig

WORKDIR /mlrun

COPY ./requirements.txt ./
COPY ./extras-requirements.txt ./extras-requirements.txt
COPY ./dockerfiles/mlrun/requirements.txt ./mlrun-image-requirements.txt

RUN python -m pip install \
    -r requirements.txt \
    -r extras-requirements.txt \
    -r mlrun-image-requirements.txt

# if we give all the requirements together pip resolver go nuts so separating between the package and the image
# requirements
COPY ./dockerfiles/base/requirements.txt ./base-image-requirements.txt
RUN python -m pip install \
    -r base-image-requirements.txt

COPY ./dockerfiles/models/requirements.txt ./models-image-requirements.txt
RUN python -m pip install \
    -r models-image-requirements.txt

COPY ./dockerfiles/models-gpu/requirements.txt ./models-gpu-image-requirements.txt
RUN python -m pip install \
    -r models-gpu-image-requirements.txt

COPY . .
RUN python -m pip install .[complete]
