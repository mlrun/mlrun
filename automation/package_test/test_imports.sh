#!/usr/bin/env bash
set -o errexit


test_import () {
    echo "Testing import: python=$PYTHON_VERSION extra=$1 import=$2"
    # Create an empty environment
    virtualenv venv
    source venv/bin/activate
    pip install ."$1"
    python -c "$2"
    deactivate
    rm -rf venv
}

test_import ""                    "import mlrun"
test_import "api"                 "import mlrun.api.main"
test_import "dask"                "from dask.distributed import Client"
test_import "v3io"                "import mlrun.datastore.v3io"
test_import "s3"                  "import mlrun.datastore.s3"
test_import "azure-blob-storage"  "import mlrun.datastore.azure_blob"
