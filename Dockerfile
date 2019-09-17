FROM python:3.6
WORKDIR /mlrun
COPY . .
RUN python setup.py install &&\
    pip install kfp &&\
    pip install kubernetes==10.0.0

