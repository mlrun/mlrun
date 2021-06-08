#!/usr/bin/env bash

DEMOS_REF=${DEMOS_REF}
TUTORIALS_DIR=../../docs/tutorial


declare -a tutorial_notebooks=(
    "01-mlrun-basics.ipynb"
    "02-model-training.ipynb"
    "03-model-serving.ipynb"
)

declare -a tutorial_notebook_assets=(
  "images/func-schedule.jpg"
  "images/jobs.png"
  "images/kubeflow-pipeline.png"
  "images/nuclio-deploy.png"
)

echo "Fetching tutorial notebooks from mlrun/demos/${DEMOS_REF} ..."
for notebook_file in "${tutorial_notebooks[@]}"
do
  curl -sSL https://raw.githubusercontent.com/mlrun/demos/${DEMOS_REF}/getting-started-tutorial/${notebook_file} --output ${TUTORIALS_DIR}/${notebook_file}
done

echo "Fetching tutorial notebooks assets mlrun/demos/${DEMOS_REF} ..."
mkdir -p ${TUTORIALS_DIR}/images
for notebook_asset in "${tutorial_notebook_assets[@]}"
do
  curl -sSL https://raw.githubusercontent.com/mlrun/demos/${DEMOS_REF}/getting-started-tutorial/${notebook_asset} --output ${TUTORIALS_DIR}/${notebook_asset}
done

echo "Fixing links to local docs in ${TUTORIALS_DIR} ..."

# escape the url (similar to DEMOS_REF but with a slight change "release/" -> "release-"
DEMOS_REF_CHANGED=$(echo "${DEMOS_REF/release\//release-}")

BASE_DOCS_URL="https://mlrun.readthedocs.io/en/${DEMOS_REF_CHANGED}"


echo "Replacing ${BASE_DOCS_URL}"

# replace BASE_DOCS_URL with ../
sed -i '' "s#${BASE_DOCS_URL}#..#g" ${TUTORIALS_DIR}/*.ipynb

# replace .html ext with .md
sed -i '' -E "s#\.\.(.+)\.html#\.\.\1\.md#g" ${TUTORIALS_DIR}/*.ipynb

echo "DONE"