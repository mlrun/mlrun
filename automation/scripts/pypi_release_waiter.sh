#!/usr/bin/env bash
set -e

released_version=$1
while true ; do
  latest_version="$(curl -sf https://pypi.org/pypi/mlrun/json | jq -r '.info.version')"
  if [ "$latest_version" = "$released_version" ]; then
    echo "Version released: $latest_version"
    break;
  else
    echo "Version not released yet. Sleeping and retrying. latest version=$latest_version waiting for version=$released_version"
  fi;
  sleep 60;
done;
