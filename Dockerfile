FROM python:3.7-slim
WORKDIR /mlrun
COPY . .
RUN python setup.py install

run pip install pandas
run pip install pyarrow
run pip install v3io_frames --upgrade

