FROM jupyter/scipy-notebook

COPY --chown=$NB_UID:$NB_GID . /home/jovyan/tmp
RUN cd /home/jovyan/tmp && pip install ".[api]" && cd /home/jovyan && rm -rf tmp

COPY --chown=$NB_UID:$NB_GID ./README.md /home/jovyan/
COPY --chown=$NB_UID:$NB_GID ./examples /home/jovyan/examples/
RUN git clone https://github.com/mlrun/demos.git /home/jovyan/demos
RUN git clone https://github.com/mlrun/functions.git /home/jovyan/functions

RUN mkdir data

ENV MLRUN_DBPATH=http://localhost:8080
ENV MLRUN_ARTIFACT_PATH=/home/jovyan/data
ENV JUPYTER_ENABLE_LAB=yes
ENV MLRUN_HTTPDB__DATA_VOLUME=/home/jovyan/data
ENV MLRUN_HTTPDB__DSN='sqlite:////home/jovyan/data/mlrun.db?check_same_thread=false'

CMD mlrun db & start-notebook.sh \
    --ip="0.0.0.0" \
    --port=8888 \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --NotebookApp.default_url="/lab"
