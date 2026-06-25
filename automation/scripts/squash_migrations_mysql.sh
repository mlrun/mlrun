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

# MLRUN_SQUASH_MODE selects what to squash (default: root):
#   root - squash from root up to and including MLRUN_SQUASH_REVISION (the target)
#   head - squash everything after MLRUN_SQUASH_REVISION (the base) up to the head
MLRUN_SQUASH_MODE="${MLRUN_SQUASH_MODE:-root}"

missing_vars=false

if [ -z "$MLRUN_MIGRATION_MESSAGE" ]; then
	echo "Environment variable MLRUN_MIGRATION_MESSAGE not set"
	missing_vars=true
fi

if [ -z "$MLRUN_SQUASH_REVISION" ]; then
	echo "Environment variable MLRUN_SQUASH_REVISION not set"
	missing_vars=true
fi

if [ "$MLRUN_SQUASH_MODE" != "root" ] && [ "$MLRUN_SQUASH_MODE" != "head" ]; then
	echo "Environment variable MLRUN_SQUASH_MODE must be 'root' or 'head' (got '${MLRUN_SQUASH_MODE}')"
	missing_vars=true
fi

if [ "$missing_vars" = true ]; then
	echo "Usage: MLRUN_MIGRATION_MESSAGE=<message> MLRUN_SQUASH_REVISION=<revision_id> [MLRUN_SQUASH_MODE=root|head] MLRUN_MYSQL_IMAGE=<image> $0"
	exit 1
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# shellcheck source=_mysql_docker_lib.sh
source "${SCRIPT_DIR}/_mysql_docker_lib.sh"

trap _mysql_cleanup SIGHUP SIGINT SIGTERM EXIT

_mysql_full_setup

cd "${_MYSQL_ROOT_DIR}/server/py/services/api"

python "${SCRIPT_DIR}/squash_migrations.py" "${MLRUN_SQUASH_MODE}" "${MLRUN_SQUASH_REVISION}" "${MLRUN_MIGRATION_MESSAGE}"
