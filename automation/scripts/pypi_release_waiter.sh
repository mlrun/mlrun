#!/usr/bin/env bash
set -e

version=$1
version=${version#"v"}
echo "Waiting for version to be released on Pypi. version:$version"
while true ; do
  latest_version="$(curl -sf https://pypi.org/pypi/mlrun/json | jq -r '.info.version')"
  if [ "$latest_version" = "$version" ]; then
    echo "Version released: $latest_version"
    break;
  else
    echo "Version not released yet. Sleeping and retrying. latest version=$latest_version waiting for version=$version"
  fi;
  sleep 60;
done;
