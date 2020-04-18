#!/bin/bash

set -e

args=("$@")

export REPO=${args[0]}
export PREFIX=${args[1]}
export MLRUN_COMMIT=${args[2]}
export NEW_TAG=${args[3]}
export PYTHON_VER=${args[4]}

# we may end up with mlrun/mlrun, mlrun/models, mlrun/serving
# and a couple of other utility images
# currently only builds the cpu `base` and `models`
for IMAGE in 'base' 'models' 'models-gpu' 'hvd' 'hvd-gpu' 'serving'
do
    docker build \
        -f $IMAGE/Dockerfile \
        --build-arg MLRUN_TAG=$MLRUN_COMMIT \
        --build-arg REPO=$REPO \
        --build-arg PREFIX=$PREFIX \
        --build-arg PYTHON_VER=$PYTHON_VER \
        -t $REPO/$PREFIX-$IMAGE:$MLRUN_COMMIT .

    # docker push $REPO/$PREFIX-$IMAGE:$MLRUN_COMMIT
    docker tag $REPO/$PREFIX-$IMAGE:$MLRUN_COMMIT $REPO/$PREFIX-$IMAGE:$NEW_TAG
    docker push $REPO/$PREFIX-$IMAGE:$NEW_TAG
done