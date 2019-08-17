FROM python:3.7-slim
WORKDIR /mlrun
COPY . .
RUN python setup.py install &&\
    pip install pyarrow &&\
    pip install v3io_frames

