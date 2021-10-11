#!/bin/bash
set -e

version=$1
version=${version#"v"}
echo "Waiting for version to be released on Pypi. version:$version"
while true ; do
  released_versions="$(curl -sf https://pypi.org/pypi/mlrun/json | jq -r '.releases | keys | join(",")')"
  if [[ "$released_versions" == *"$version"* ]]; then
    echo "Version released: $version"
    break;
  else
    echo "Version not released yet. Sleeping and retrying. waiting for version=$version released_versions=$released_versions"
  fi;
  sleep 60;
done;
