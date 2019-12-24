ARG UBUNTU_VERSION=16.04
FROM ubuntu:${UBUNTU_VERSION}
# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
ENV TENSORFLOW_VERSION=1.13.1
ENV PYTORCH_VERSION=1.2.0

#ENV MXNET_URL=https://s3-us-west-2.amazonaws.com/mxnet-python-packages-gcc5/mxnet_cu90_gcc5-1.4.0-py2.py3-none-manylinux1_x86_64.whl
# Python 2.7 or 3.5 is supported by Ubuntu Xenial out of the box
ARG python=3.6
ENV PYTHON_VERSION=${python}

# Add python 3.6 repository
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update

RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv
RUN python3.6 -m pip install pip --upgrade
RUN pip install --upgrade setuptools

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libjpeg-dev \
        libpng-dev

# Update gcc/g++ to v4.9
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt-get update -qq
RUN apt-get install -qq gcc-4.9 g++-4.9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 90
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 90
RUN update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-4.9 90
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py
# Install TensorFlow, Keras, PyTorch and MXNet
RUN pip install matplotlib pandas scikit-learn 'numpy<1.15.0' tensorflow==${TENSORFLOW_VERSION} keras h5py torch==${PYTORCH_VERSION} torchvision #${MXNET_URL}

# Install mlrun @ development

RUN pip install --upgrade pip
# RUN pip install aiohttp==3.6.1
#RUN pip install git+https://github.com/mlrun/mlrun.git@v40-fixes-and-docs
#RUN pip install mlrun

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda-10.0/targets/x86_64-linux/lib/stubs && \
    HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod && \
    ldconfig

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

RUN pip install mlrun
RUN useradd -ms /bin/bash horovod
#RUN chown -R horovod:horovod /examples
USER horovod
