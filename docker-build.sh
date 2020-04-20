#!/bin/bash

set -e

args=("$@")

export REPO=${args[0]}
export PREFIX=${args[1]}
export MLRUN_COMMIT=${args[2]}
export NEW_TAG=${args[3]}
export PYTHON_VER_ML=${args[4]}
export PYTHON_VER_CORE=${args[5]}

for IMAGE in 'base' 'models' 'models-gpu' 'serving'
do
    docker build \
        -f ./dockerfiles/$IMAGE/Dockerfile \
        --build-arg MLRUN_TAG=$MLRUN_COMMIT \
        --build-arg REPO=$REPO \
        --build-arg PREFIX=$PREFIX \
        --build-arg PYTHON_VER=$PYTHON_VER_ML \
        -t $REPO/$PREFIX-$IMAGE:$NEW_TAG .

    docker push $REPO/$PREFIX-$IMAGE:$NEW_TAG
done

for IMAGE in 'dask' 'httpd' 'test'
do
    docker build \
        -f ./dockerfiles/$IMAGE/Dockerfile \
        --build-arg MLRUN_TAG=$MLRUN_COMMIT \
        --build-arg REPO=$REPO \
        --build-arg PYTHON_VER=$PYTHON_VER_CORE \
        -t $REPO/$IMAGE:$NEW_TAG .

    docker push $REPO/$IMAGE:$NEW_TAG
done
