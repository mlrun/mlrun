FROM jupyter/scipy-notebook:python-3.8.8

USER root
RUN apt-get update && apt-get install --no-install-recommends -y \
  graphviz \
 && rm -rf /var/lib/apt/lists/*
USER $NB_UID

ENV PIP_NO_CACHE_DIR=1

RUN python -m pip install --upgrade pip~=21.2.0

WORKDIR $HOME

COPY --chown=$NB_UID:$NB_GID ./README.md $HOME
COPY --chown=$NB_UID:$NB_GID ./examples/mlrun_basics.ipynb $HOME
COPY --chown=$NB_UID:$NB_GID ./docs/quick-start.ipynb $HOME
COPY --chown=$NB_UID:$NB_GID ./docs/_static/images/mlrun-quick-start $HOME/_static/images/mlrun-quick-start

ARG MLRUN_CACHE_DATE=initial
RUN git clone --branch hackathon https://github.com/mlrun/demos.git $HOME/demos
RUN git clone --branch master https://github.com/mlrun/functions.git $HOME/functions

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

ENV MLRUN_ARTIFACT_PATH=$HOME/data \
    JUPYTER_ENABLE_LAB=yes \
    MLRUN_HTTPDB__DATA_VOLUME=$HOME/data \
    MLRUN_HTTPDB__DSN='sqlite:////home/jovyan/data/mlrun.db?check_same_thread=false' \
    MLRUN_HTTPDB__LOGS_PATH=$HOME/data/logs \
    MLRUN_PVC_MOUNT=nfsvol:/home/jovyan/data

# run the mlrun db (api) and the notebook in parallel
CMD python -m mlrun db & MLRUN_DBPATH=http://localhost:8080 start-notebook.sh \
    --ip="0.0.0.0" \
    --port=8888 \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --NotebookApp.default_url="/lab"
