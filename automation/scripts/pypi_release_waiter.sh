# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#!/bin/bash
set -e

version=$1
version=${version#"v"}
echo "Waiting for version to be released on Pypi. version:$version"
while true ; do
  released_versions="$(curl -sf https://pypi.org/pypi/mlrun/json | jq -r '.releases | keys | join(",")')"
  if [[ "$released_versions" == *,"$version",* ]]; then
    echo "Version released: $version"
    break;
  else
    echo "Version not released yet. Sleeping and retrying. waiting for version=$version released_versions=$released_versions"
  fi;
  sleep 60;
done;
