#! /usr/bin/env bash

# Execute docker prestart
./docker-prestart.sh

cd /mlrun || exit

uvicorn mlrun.api.main:app --port "$PORT" --host 0.0.0.0



