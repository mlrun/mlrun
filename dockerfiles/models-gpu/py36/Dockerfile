ARG CUDA_VER=10.1

FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

LABEL maintainer="yashab@iguazio.com"
LABEL org="iguazio.com"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && \
    apt-mark hold libcublas-dev libcublas10 && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    bzip2 \
    cmake \
    curl \
    git \
    ibverbs-providers \
    libboost-dev \
    libboost-filesystem-dev \
    libboost-system-dev \
    libc6 \
    libglib2.0-0 \
    libibverbs1\
    libjpeg-dev \
    libnvinfer-plugin6=6.0.1-1+cuda10.1 \
    libnvinfer6=6.0.1-1+cuda10.1 \
    libpng-dev \
    librdmacm1 \
    libreadline-dev \
    libreadline7 \
    libsm6 \
    libxext6 \
    libxrender1 \
    mercurial \
    openssh-client \
    openssh-server \
    subversion \
    wget && \
    apt install -y ca-certificates  && \
    update-ca-certificates --fresh && \
    apt clean && \
    apt autoremove && \
    rm -rf /var/lib/apt/lists/*

ENV SSL_CERT_DIR /etc/ssl/certs

RUN mkdir -p /var/run/sshd && \
    # Allow OpenSSH to talk to containers without asking for confirmation
    cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config && \
    svn checkout https://github.com/horovod/horovod/trunk/examples && \
    rm -rf /examples/.svn

ARG OMPI=4.0.3

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    curl -o openmpi-${OMPI}.tar.gz https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-${OMPI}.tar.gz && \
    tar zxf openmpi-${OMPI}.tar.gz && \
    cd openmpi-${OMPI} && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j`nproc` all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi    

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/installconda.sh && \
    /bin/bash ~/installconda.sh -b -p /opt/conda && \
    rm ~/installconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN conda config --add channels plotly && \
    conda config --add channels bokeh && \
    conda config --add channels conda-forge && \
    conda config --add channels anaconda && \
    conda update -n base conda && \
    conda install -n base python=3.6 \
    bokeh \
    cmake \
    cudatoolkit=10.1 \
    cython \
    fsspec \
    lifelines \
    matplotlib \
    numba \
    numpy \
    pandas \
    plotly \
    plotly-orca \
    pyarrow \
    pytest \
    scikit-build \
    scikit-learn \
    scikit-optimize \
    scikit-plot \
    scipy \
    seaborn \
    tensorflow-gpu \
    wrapt

RUN conda install -n base -c rapidsai -c nvidia \
    -c anaconda -c conda-forge -c defaults rapids=0.12

RUN conda install -n base -c pytorch pytorch torchvision cudatoolkit=10.1

RUN conda clean -aqy

# TODO: MAKEFLAGS="-j1" work around some transient concurrency problem with installing horovod remove it when
    #   possible (should be safe to remove if it works ~5 times without it)
RUN ldconfig /usr/local/cuda-10.1/targets/x86_64-linux/lib/stubs && \
    MAKEFLAGS="-j1" HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 \
    python -m pip install horovod && \
    ldconfig

WORKDIR /mlrun

# remove dask verify version mismatch warnings
COPY ./dockerfiles/models-gpu/requirements.txt ./

RUN python -m pip install -r requirements.txt

COPY . .
RUN python -m pip install .

ARG MLRUN_MLUTILS_GITHUB_TAG=development
ARG MLRUN_MLUTILS_CACHE_DATE=initial
RUN python -m pip install \
    git+https://github.com/mlrun/mlutils.git@${MLRUN_MLUTILS_GITHUB_TAG}
