#! /usr/bin/env bash

# Execute docker prestart
/mlrun/scripts/api/docker-prestart.sh

# shellcheck disable=SC2154
uvicorn mlrun.api.main:app --port "$MLRUN_httpdb__port" --host 0.0.0.0



