FROM jupyter/scipy-notebook
RUN pip install mlrun[api]
COPY ./README.md /home/jovyan/
COPY ./examples /home/jovyan/examples/

USER jovyan
RUN git clone https://github.com/mlrun/demos.git /home/jovyan/demos
RUN git clone https://github.com/mlrun/functions.git /home/jovyan/functions

USER root

ENV MLRUN_DBPATH=http://localhost:8080
ENV MLRUN_ARTIFACT_PATH=/home/jovyan/data
ENV JUPYTER_ENABLE_LAB=yes
ENV MLRUN_HTTPDB__DATA_VOLUME=/home/jovyan/data

CMD start-notebook.sh \
CMD mlrun db & start-notebook.sh \
    --ip="0.0.0.0" \
    --port=8888 \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --NotebookApp.default_url="/lab"

