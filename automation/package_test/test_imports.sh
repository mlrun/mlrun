#!/usr/bin/env bash
set -o errexit


test_import () {
    echo "Testing import: python=$PYTHON_VERSION extra=$1 import=$2"
    # Create an empty environment
    python -m pip install virtualenv
    virtualenv venv
    source venv/bin/activate
    pip install ."$1"
    python -c "$2"
    deactivate
    rm -rf venv
}

basic_test="import mlrun"
api_test="import mlrun.api.main"
s3_test="import mlrun.datastore.s3"
azure_blob_storage_test="import mlrun.datastore.azure_blob"

test_import ""                      "$basic_test"
# API works only with python 3.7 and above
if [ "$PYTHON_VERSION" != "3.6" ]
  then
    test_import "[api]"                 "$basic_test; $api_test"
    test_import "[complete-api]"        "$basic_test; $api_test; $s3_test; $azure_blob_storage_test"
fi
test_import "[s3]"                  "$basic_test; $s3_test"
test_import "[azure-blob-storage]"  "$basic_test; $azure_blob_storage_test"
test_import "[complete]"  "$basic_test; $s3_test; $azure_blob_storage_test"
