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

test_import ""                      "import mlrun"
# API works only with python 3.7 and above
if [ "$PYTHON_VERSION" != "3.6" ]
  then
    test_import "[api]"                 "import mlrun.api.main"
fi
test_import "[dask]"                "import dask"
test_import "[v3io]"                "import mlrun.datastore.v3io"
test_import "[s3]"                  "import mlrun.datastore.s3"
test_import "[azure-blob-storage]"  "import mlrun.datastore.azure_blob"
