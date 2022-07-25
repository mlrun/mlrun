FROM jupyter/scipy-notebook:python-3.8.8

USER root
RUN apt-get update && \
    apt-get -s dist-upgrade | \
    grep "^Inst" | \
    grep -i securi | \
    awk -F " " {'print $2'} | \
    xargs apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install --no-install-recommends -y \
  graphviz \
 && rm -rf /var/lib/apt/lists/*
USER $NB_UID

ENV PIP_NO_CACHE_DIR=1

ARG MLRUN_PIP_VERSION=22.0.0
RUN python -m pip install --upgrade pip~=${MLRUN_PIP_VERSION}

WORKDIR $HOME

COPY --chown=$NB_UID:$NB_GID ./dockerfiles/jupyter/README.ipynb $HOME
COPY --chown=$NB_UID:$NB_GID ./dockerfiles/jupyter/mlrun.env $HOME
COPY --chown=$NB_UID:$NB_GID ./docs/tutorial $HOME/tutorial
COPY --chown=$NB_UID:$NB_GID ./docs/_static/images/tutorial $HOME/_static/images/tutorial
COPY --chown=$NB_UID:$NB_GID ./docs/_static/images/MLRun-logo.png $HOME/_static/images

RUN mkdir data

COPY --chown=$NB_UID:$NB_GID ./dockerfiles/jupyter/requirements.txt /tmp/requirements/jupyter-requirements.txt
COPY --chown=$NB_UID:$NB_GID ./dockerfiles/mlrun-api/requirements.txt /tmp/requirements/mlrun-api-requirements.txt
COPY --chown=$NB_UID:$NB_GID ./extras-requirements.txt /tmp/requirements/extras-requirement.txt
COPY --chown=$NB_UID:$NB_GID ./requirements.txt /tmp/requirements/requirement.txt
RUN python -m pip install \
    -r /tmp/requirements/jupyter-requirements.txt \
    -r /tmp/requirements/mlrun-api-requirements.txt \
    -r /tmp/requirements/extras-requirement.txt \
    -r /tmp/requirements/requirement.txt

COPY --chown=$NB_UID:$NB_GID . /tmp/mlrun
RUN cd /tmp/mlrun && python -m pip install ".[complete-api]"

# This will usually cause a cache miss - so keep it in the last layers
ARG MLRUN_CACHE_DATE=initial
RUN git clone --branch 1.0.x https://github.com/mlrun/demos.git $HOME/demos
RUN git clone --branch master https://github.com/mlrun/functions.git $HOME/functions

ENV JUPYTER_ENABLE_LAB=yes \
    MLRUN_HTTPDB__DATA_VOLUME=$HOME/data \
    MLRUN_HTTPDB__DSN='sqlite:////home/jovyan/data/mlrun.db?check_same_thread=false' \
    MLRUN_HTTPDB__LOGS_PATH=$HOME/data/logs \
    MLRUN_ENV_FILE=$HOME/mlrun.env

# run the mlrun db (api) and the notebook in parallel
CMD MLRUN_ARTIFACT_PATH=$HOME/data python -m mlrun db & MLRUN_DBPATH=http://localhost:8080 start-notebook.sh \
    --ip="0.0.0.0" \
    --port=8888 \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --NotebookApp.default_url="/lab"
