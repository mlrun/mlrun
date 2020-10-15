FROM jupyter/scipy-notebook

# by default COPY creates files with root permissions, so doing all the preparation with root and chown-ing everything
# in the end
USER root
COPY . /tmp/mlrun
RUN cd /tmp/mlrun && python -m pip install --no-cache-dir ".[api]" && cd /tmp && rm -rf mlrun

COPY ./dockerfiles/jupyter/requirements.txt /tmp
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

WORKDIR $HOME

COPY ./README.md $HOME
COPY ./examples $HOME/examples/

ARG MLRUN_CACHE_DATE=initial
RUN git clone https://github.com/mlrun/demos.git $HOME/demos
RUN git clone https://github.com/mlrun/functions.git $HOME/functions

RUN mkdir data

RUN chown -R $NB_UID:$NB_GID $HOME

USER $NB_UID

ENV MLRUN_DBPATH=http://localhost:8080 \
    MLRUN_ARTIFACT_PATH=$HOME/data \
    JUPYTER_ENABLE_LAB=yes \
    MLRUN_HTTPDB__DATA_VOLUME=$HOME/data \
    MLRUN_HTTPDB__DSN='sqlite:////home/jovyan/data/mlrun.db?check_same_thread=false' \
    MLRUN_HTTPDB__LOGS_PATH=$HOME/data/logs \
    MLRUN_PVC_MOUNT=nfsvol:/home/jovyan/data

# run the mlrun db (api) and the notebook in parallel
CMD mlrun db & start-notebook.sh \
    --ip="0.0.0.0" \
    --port=8888 \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --NotebookApp.default_url="/lab"
