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

get_os() {
  unameOut="$(uname -s)"
  case "${unameOut}" in
      Linux*)     os=Linux;;
      Darwin*)    os=Mac;;
      *)          os="UNKNOWN:${unameOut}"
  esac
  echo ${os}
}

SCHEMAS_DIR=../mlrun/api/proto/
SED_REGEX='s/from proto import/from \. import/g'
OS=$(get_os)
SCHEMA_FILES=$(find ../mlrun/api/proto/ -name '*pb2_grpc.py')

if [ "${OS}" == "Mac" ]; then
  sed -i '' -e "${SED_REGEX}" ${SCHEMA_FILES}
else
  sed -i -e "${SED_REGEX}" ${SCHEMA_FILES}
fi
