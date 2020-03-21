# Copyright 2020 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
FROM continuumio/miniconda3:4.7.12

MAINTAINER yashab@iguazio.com

ARG REPO
ARG PREFIX
ARG MLRUN_TAG


RUN apt update -qqq \
	&& apt -y upgrade \
	&& apt install -yq --no-install-recommends build-essential \
	graphviz \
	cmake \
	curl \
	libreadline7 \
	libreadline-dev \
	libboost-dev \
	libboost-system-dev \
	libboost-filesystem-dev \
	&& apt install -y ca-certificates \
	&& update-ca-certificates --fresh \
	&& apt clean \
	&& apt autoremove \
	&& rm -rf /var/lib/apt/lists/*

ENV SSL_CERT_DIR /etc/ssl/certs

RUN conda update conda && \
	conda config --add channels intel && \
	conda config --add channels conda-forge && \
	conda init bash && \
	conda install -n base intelpython3_core python=3 h5py joblib \
	dill cloudpickle python-blosc lz4 msgpack-python==0.6.2 'fsspec>=0.6.2' \
	tini==0.18.0 cytoolz nomkl python-graphviz 

# ml/ai pipelines
RUN conda install -n base kfp

RUN python -m pip install  --ignore-installed git+https://github.com/mlrun/mlrun.git@${MLRUN_TAG}

RUN python -m pip uninstall -y pyarrow && python -m pip install pyarrow

RUN python -m pip install -U gnureadline

COPY base/prepare.sh /usr/bin/prepare.sh

RUN mkdir /opt/app

ENTRYPOINT ["tini", "-g", "--", "/usr/bin/prepare.sh"]
