FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

LABEL maintainer="yashab@iguazio.com"
LABEL org="iguazio.com"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --allow-change-held-packages \
        build-essential \
        cmake \
        git \
        curl \
        nano \
        wget \
        bzip2 \
        libglib2.0-0 \
        libxext6 \
        libsm6 \
        libxrender1 \
        mercurial \
        subversion \
        libreadline7 \
        libreadline-dev \
        libboost-dev \
        libboost-system-dev \
        libboost-filesystem-dev \
        libjpeg-dev \
        libpng-dev \
        librdmacm1 \
        libibverbs1\
        ibverbs-providers && \
    apt install -y ca-certificates  && \
    update-ca-certificates --fresh && \
    apt clean && \
    apt autoremove && \
    rm -rf /var/lib/apt/lists/*

ENV SSL_CERT_DIR /etc/ssl/certs

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/installconda.sh && \
    /bin/bash ~/installconda.sh -b -p /opt/conda && \
    rm ~/installconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN conda config --add channels conda-forge && \
    conda config --add channels anaconda && \
    conda update -n base conda && \
    conda install -n base cudatoolkit=10.1 \
                          cmake \
                          tensorflow \
                          pandas \
                          numpy \
                          scipy \
                          scikit-learn \
                          scikit-optimize \
                          scikit-build \
                          matplotlib \
                          seaborn \
                          pytest \
                          pyarrow \
                          numba \
                          cython \
                          wrapt \
                          dask[complete] \
                          distributed \
                          dask-ml \
                          dask-kubernetes \
                          dask-xgboost \
                          fsspec \
                          bokeh


RUN conda install -n base -c rapidsai -c nvidia -c anaconda -c conda-forge -c defaults rapids=0.12

RUN conda install pytorch torchvision -c pytorch

RUN find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

RUN python -m pip install  

RUN python -m pip install --no-cache-dir \
        yellowbrick \
        gnureadline \
        pyod \
        urllib3==1.24.3 \
        git+https://github.com/scikit-multiflow/scikit-multiflow \
        git+https://github.com/mlrun/mlrun.git@development

# xgboost
RUN mkdir /repos && \
    cd repos && \
    git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && \
    git checkout v1.0.2 && \
    mkdir build && \
    cd build && \
    cmake .. -DUSE_CUDA=ON && \
    make -j$(nproc) && \
    make install && \
    cd ../python-package && \
    python -m pip install --no-cache-dir -e .

RUN python  -m pip install git+https://github.com/yjb-ds/mlutils.git

WORKDIR "/examples"
