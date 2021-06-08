#!/usr/bin/env bash

set -e

# this is intended to be run from the repo root
ROOT_DIR=$(pwd)
MLRUN_DEMOS_REF=${MLRUN_DEMOS_REF}
TUTORIALS_DIR=${ROOT_DIR}/docs/tutorial


declare -a tutorial_notebooks=(
    "01-mlrun-basics.ipynb"
    "02-model-training.ipynb"
    "03-model-serving.ipynb"
    "04-pipeline.ipynb"
)

declare -a tutorial_notebook_assets=(
  "images/func-schedule.jpg"
  "images/jobs.png"
  "images/kubeflow-pipeline.png"
  "images/nuclio-deploy.png"
)

echo "Fetching tutorial notebooks from mlrun/demos/${MLRUN_DEMOS_REF} ..."
for notebook_file in "${tutorial_notebooks[@]}"
do
  curl -sSL https://raw.githubusercontent.com/mlrun/demos/${MLRUN_DEMOS_REF}/getting-started-tutorial/${notebook_file} --output ${TUTORIALS_DIR}/${notebook_file}
done

echo "Fetching tutorial notebooks assets mlrun/demos/${MLRUN_DEMOS_REF} ..."
mkdir -p ${TUTORIALS_DIR}/images
for notebook_asset in "${tutorial_notebook_assets[@]}"
do
  curl -sSL https://raw.githubusercontent.com/mlrun/demos/${MLRUN_DEMOS_REF}/getting-started-tutorial/${notebook_asset} --output ${TUTORIALS_DIR}/${notebook_asset}
done

echo "Fixing links to local docs in ${TUTORIALS_DIR} ..."

# escape the url (similar to MLRUN_DEMOS_REF but with a slight change "release/" -> "release-"
MLRUN_DEMOS_REF_CHANGED=$(echo "${MLRUN_DEMOS_REF/release\//release-}")

BASE_DOCS_URL="https://mlrun.readthedocs.io/en/${MLRUN_DEMOS_REF_CHANGED}"


echo "Replacing ${BASE_DOCS_URL}"

# replace BASE_DOCS_URL with ../
sed -i '' "s#${BASE_DOCS_URL}#..#g" ${TUTORIALS_DIR}/*.ipynb

# replace .html ext with .md
sed -i '' -E "s#\.\.(.+)\.html#\.\.\1\.md#g" ${TUTORIALS_DIR}/*.ipynb

echo "DONE"