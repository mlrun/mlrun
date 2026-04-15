#!/bin/bash
# Copyright 2026 Iguazio
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

set -e

if [ -z "$MLRUN_SQUASH_REVISION" ]; then
	echo "Environment variable MLRUN_SQUASH_REVISION not set"
	echo "Usage: MLRUN_SQUASH_REVISION=<revision_id> MLRUN_MYSQL_IMAGE=<image> $0"
	exit 1
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# shellcheck source=_mysql_docker_lib.sh
source "${SCRIPT_DIR}/_mysql_docker_lib.sh"

trap _mysql_cleanup SIGHUP SIGINT SIGTERM EXIT

_mysql_full_setup

cd "${_MYSQL_ROOT_DIR}/server/py/services/api"

python "${SCRIPT_DIR}/squash_migrations.py" "${MLRUN_SQUASH_REVISION}"
ruff format migrations/versions/${MLRUN_SQUASH_REVISION}_*.py
ruff check --fix migrations/versions/${MLRUN_SQUASH_REVISION}_*.py
