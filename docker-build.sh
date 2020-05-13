#!/bin/bash

set -e

args=("$@")

export REPO=${args[0]}
export PREFIX=${args[1]}
export MLRUN_COMMIT=${args[2]}
export NEW_TAG=${args[3]}
export PYTHON_VER_ML=${args[4]}
export PYTHON_VER_CORE=${args[5]}

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
        -t $REPO/$PREFIX-$IMAGE:$NEW_TAG-py"${PYTHON_VER_CORE//.}" .

    docker push $REPO/$PREFIX-$IMAGE:$NEW_TAG-py"${PYTHON_VER_CORE//.}"
done

for IMAGE in 'mlrun-api' 'test'
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