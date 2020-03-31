FROM jupyter/scipy-notebook
RUN pip install mlrun[dask]
copy ./README.md /home/jovyan/
copy ./examples /home/jovyan/examples/

ENV MLRUN_ARTIFACT_PATH=/home/jovyan/data
ENV JUPYTER_ENABLE_LAB=yes

CMD start-notebook.sh \
    --ip="0.0.0.0" \
    --port=8888 \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --NotebookApp.default_url="/lab"
