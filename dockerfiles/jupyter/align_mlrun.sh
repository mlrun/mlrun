#!/bin/bash
CLIENT_MLRUN_VERSION=$(pip show mlrun | grep Version | awk '{print $2}')
SERVER_MLRUN_VERSION=$(curl -s http://mlrun-api:8080/api/v1/client-spec | python3 -c "import sys, json; print(json.load(sys.stdin)['version'])")
if [ "${CLIENT_MLRUN_VERSION}" = "${SERVER_MLRUN_VERSION}" ] || [ "${CLIENT_MLRUN_VERSION}" = "${SERVER_MLRUN_VERSION//-}" ]; then
  echo "Both server & client are aligned (${CLIENT_MLRUN_VERSION})."
else
  if [ ${CLIENT_MLRUN_VERSION} ]; then
    echo "Server ${SERVER_MLRUN_VERSION} & client ${CLIENT_MLRUN_VERSION} are unaligned."
    echo "Updating client..."
    pip uninstall -y mlrun
  fi
  pip install mlrun[complete]==${SERVER_MLRUN_VERSION}
fi
