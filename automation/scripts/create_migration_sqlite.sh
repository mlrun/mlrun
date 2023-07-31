#!/bin/bash
# Copyright 2023 Iguazio
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

set -e

if [ ! -v  MLRUN_MIGRATION_MESSAGE ]; then
	echo "Environment variable MLRUN_MIGRATION_MESSAGE not set"
	exit 1
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR="${SCRIPT_DIR}/../.."

export MLRUN_HTTPDB__DSN="sqlite:///${ROOT_DIR}/mlrun/api/migrations_sqlite/mlrun.db?check_same_thread=false"

alembic -c "${ROOT_DIR}/mlrun/api/alembic.ini" upgrade head
alembic -c "${ROOT_DIR}/mlrun/api/alembic.ini" revision --autogenerate -m "$(MLRUN_MIGRATION_MESSAGE)"
