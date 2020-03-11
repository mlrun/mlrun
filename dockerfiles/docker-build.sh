#!/bin/bash

set -e

args=("$@")

export REPO=${args[0]}
export PREFIX=${args[1]}
export MLRUN_COMMIT=${args[2]}
export NEW_TAG=${args[3]}

for IMAGE in 'base' 'models' 'dask' 'serving'
do
    docker build \
        -f $IMAGE/Dockerfile \
        --build-arg MLRUN_TAG=$MLRUN_COMMIT \
        --build-arg REPO=$REPO \
        --build-arg PREFIX=$PREFIX \
        -t $REPO/$PREFIX-$IMAGE:$MLRUN_COMMIT .

    docker push $REPO/$PREFIX-$IMAGE:$MLRUN_COMMIT
    docker tag $REPO/$PREFIX-$IMAGE:$MLRUN_COMMIT $REPO/$PREFIX-$IMAGE:$NEW_TAG
    docker push $REPO/$PREFIX-$IMAGE:$NEW_TAG
done