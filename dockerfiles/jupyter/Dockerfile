# Copyright 2023 Iguazio
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

ARG MLRUN_PYTHON_VERSION=3.11
ARG MLRUN_PIP_VERSION=26.0
ARG MLRUN_UV_IMAGE=ghcr.io/astral-sh/uv:latest
ARG MLRUN_CACHE_DATE=initial

FROM ${MLRUN_UV_IMAGE} AS uv-image
FROM quay.io/jupyter/scipy-notebook:python-${MLRUN_PYTHON_VERSION}
ARG MLRUN_PYTHON_VERSION
ARG MLRUN_PIP_VERSION
USER root

# System dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install --no-install-recommends -y \
      graphviz \
      curl \
      apt-transport-https \
      unzip && \
    rm -rf /var/lib/apt/lists/*

# Download and install kubectl
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    chmod +x kubectl && \
    mv kubectl /usr/local/bin/

# Create and activate mlrun conda environment
RUN conda create -y -n mlrun python=${MLRUN_PYTHON_VERSION} && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate mlrun" >> ~/.bashrc && \
    chown -R $NB_UID:$NB_GID /opt/conda/envs/mlrun

USER $NB_UID

# Ensure uv uses mlrun conda env
ENV PATH="/opt/conda/envs/mlrun/bin:$PATH"

RUN python -m pip install --upgrade pip~=${MLRUN_PIP_VERSION} && \
    conda update --all --yes

# Add Jupyter kernel for mlrun environment
RUN conda install -n mlrun ipykernel -y && \
    python -m ipykernel install --user --name=mlrun --display-name "mlrun"

WORKDIR $HOME

COPY --chown=$NB_UID:$NB_GID ./docs/tutorials $HOME/tutorials
COPY --chown=$NB_UID:$NB_GID ./docs/_static/images/MLRun-logo.png $HOME/_static/images/MLRun-logo.png
COPY --chown=$NB_UID:$NB_GID ./dockerfiles/jupyter/README.ipynb $HOME
COPY --chown=$NB_UID:$NB_GID ./dockerfiles/jupyter/mlrun.env $HOME
COPY --chown=$NB_UID:$NB_GID ./dockerfiles/jupyter/mlce-start.sh /usr/local/bin/mlce-start.sh
COPY --chown=$NB_UID:$NB_GID ./dockerfiles/jupyter/align_mlrun.sh $HOME
COPY --chown=$NB_UID:$NB_GID ./automation/scripts/get_demos.py $HOME
COPY --chown=$NB_UID:$NB_GID ./automation/scripts/demos_config.json $HOME

ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1

# removing unsupported files
RUN rm -rf $HOME/tutorials/genai-01-basic-tutorial.ipynb $HOME/tutorials/genai-02-model-monitor-llm.ipynb \
    $HOME/tutorials/genai-03-vector-db.ipynb

# no-deps to ignore existing dependencies, just add the locked requirements
# mainly because of how the jupyter image is built with conda
# can be removed once we have newer jupyter image
RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,id=pip-${MLRUN_PYTHON_VERSION},target=/root/.cache/uv \
    --mount=type=bind,source=dockerfiles/jupyter/locked-requirements.txt,target=locked-requirements.txt \
    CONDA_PREFIX=/opt/conda/envs/mlrun uv pip install --no-deps --require-hashes -r locked-requirements.txt --python-version ${MLRUN_PYTHON_VERSION}

# ensure we have the latest source code to be installed
COPY --chown=$NB_UID:$NB_GID . /tmp/mlrun

RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,id=pip-${MLRUN_PYTHON_VERSION},target=/root/.cache/uv \
    cd /tmp/mlrun && CONDA_PREFIX=/opt/conda/envs/mlrun uv pip install '.[complete]' --python-version ${MLRUN_PYTHON_VERSION} && uv cache clean


# Switch to root to clean up files that may have been written with root permissions
USER root

# Cleanup source code is not needed anymore
# Remove system ensurepip Python runtimes
RUN rm -rf /tmp/mlrun \
    && rm -rf $HOME/.cache \
    && rm -rf /opt/conda/envs/mlrun/lib/python${MLRUN_PYTHON_VERSION}/ensurepip \
    && rm -rf /opt/conda/lib/python${MLRUN_PYTHON_VERSION}/ensurepip \
    && rm -rf /usr/local/lib/python${MLRUN_PYTHON_VERSION}/ensurepip

USER $NB_UID

# This will usually cause a cache miss - so keep it in the last layers
ARG MLRUN_CACHE_DATE

# Download demos aligned with installed MLRun version using get_demos.py
RUN python -m pip install --no-cache-dir requests packaging tqdm && \
    MLVER=$(python -c "import mlrun; print(mlrun.__version__)") && \
    python get_demos.py "$MLVER" --dest "$HOME/demos" && \
    rm -rf "$HOME/demos_config.json" "$HOME/get_demos.py"

ENV JUPYTER_ENABLE_LAB=yes \
    MLRUN_HTTPDB__DATA_VOLUME=$HOME/data \
    MLRUN_HTTPDB__DSN='sqlite:////home/jovyan/data/mlrun.db?check_same_thread=false' \
    MLRUN_HTTPDB__LOGS_PATH=$HOME/data/logs \
    MLRUN_ENV_FILE=$HOME/mlrun.env \
    MLRUN_HTTPDB__PORT=8080

# backup home since it will be deleted when using pvc
RUN mkdir data && tar -cvf /tmp/basehome.tar $HOME

# use tini as entrypoint to allow signal handling
# and avoid zombie processes
ENTRYPOINT ["tini", "--"]

#This CMD only executes when running directly on docker, not k8s
CMD echo "1" > "${HOME}/.intdata" && /usr/local/bin/mlce-start.sh
