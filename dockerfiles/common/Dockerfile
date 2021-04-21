ARG MLRUN_BASE_IMAGE=mlrun/mlrun:unstable-core

FROM ${MLRUN_BASE_IMAGE}

COPY . .
RUN python -m pip install .[complete]
