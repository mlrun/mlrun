#!/bin/bash

set -e

args=("$@")

export REPO=mlrun
export PREFIX=ml
export MLRUN_COMMIT=development
export NEW_TAG=0.4.7
export PYTHON_VER_ML=3.7
export PYTHON_VER_CORE=3.6

# export DOCKER_BUILDKIT=1

for IMAGE in 'base' 'models' 'models-gpu'
do
    docker build  \
        -f ./dockerfiles/$IMAGE/Dockerfile \
        --build-arg MLRUN_TAG=$MLRUN_COMMIT \
        --build-arg REPO=$REPO \
        --build-arg PREFIX=$PREFIX \
        --build-arg NEW_TAG=$NEW_TAG \
        --build-arg PYTHON_VER=$PYTHON_VER_ML \
        -t $REPO/$PREFIX-$IMAGE:$NEW_TAG .

    docker push $REPO/$PREFIX-$IMAGE:$NEW_TAG
done

# need some python 3.6 for (legacy) package consistency 
for IMAGE in 'base' 'models' 'models-gpu'
do
    docker build \
        -f ./dockerfiles/$IMAGE/py"${PYTHON_VER_CORE//.}"/Dockerfile \
        --build-arg MLRUN_TAG=$MLRUN_COMMIT \
        --build-arg REPO=$REPO \
        --build-arg PREFIX=$PREFIX \
        --build-arg NEW_TAG=$NEW_TAG \
        --build-arg PYTHON_VER=$PYTHON_VER_CORE \
        -t yjbds/$PREFIX-$IMAGE:$NEW_TAG-py"${PYTHON_VER_CORE//.}" .

    docker push yjbds/$PREFIX-$IMAGE:$NEW_TAG-py"${PYTHON_VER_CORE//.}"
done

for IMAGE in 'dask' 'mlrun-api' 'test'
do
    docker build \ 
        -f ./dockerfiles/$IMAGE/Dockerfile \
        --build-arg MLRUN_TAG=$MLRUN_COMMIT \
        --build-arg REPO=$REPO \
        --build-arg PYTHON_VER=$PYTHON_VER_CORE \
        --build-arg NEW_TAG=$NEW_TAG \
        -t $REPO/$IMAGE:$NEW_TAG .

    docker push $REPO/$IMAGE:$NEW_TAG
done

docker build \
        --build-arg PYTHON_VER=$PYTHON_VER_CORE \
        -t $REPO/mlrun:$NEW_TAG .

docker push $REPO/mlrun:$NEW_TAG