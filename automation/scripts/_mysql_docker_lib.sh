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

# _mysql_docker_lib.sh - shared helpers for scripts that need a MySQL Docker container.
# Source this file; do not execute it directly.
#
# Required before sourcing:
#   MLRUN_MYSQL_IMAGE  - Docker image for MySQL (e.g. mysql:8.0)
#
# Configurable (optional, with defaults):
#   MLRUN_DOCKER_CONTAINER_NAME  - container name  (default: migration-db)
#   MLRUN_DOCKER_MYSQL_PORT      - host port        (default: 3306)
#   MLRUN_DOCKER_MYSQL_PASSWORD  - root password    (default: pass)
#   MLRUN_DOCKER_MYSQL_DATABASE  - database name    (default: mlrun)
#
# Exports after calling _mysql_setup_env / _mysql_full_setup:
#   MLRUN_HTTPDB__DSN  - SQLAlchemy DSN for the container
#   PYTHONPATH         - adjusted for mlrun imports

MLRUN_DOCKER_CONTAINER_NAME="${MLRUN_DOCKER_CONTAINER_NAME:-migration-db}"
MLRUN_DOCKER_MYSQL_PORT="${MLRUN_DOCKER_MYSQL_PORT:-3306}"
MLRUN_DOCKER_MYSQL_PASSWORD="${MLRUN_DOCKER_MYSQL_PASSWORD:-pass}"
MLRUN_DOCKER_MYSQL_DATABASE="${MLRUN_DOCKER_MYSQL_DATABASE:-mlrun}"

# Resolve root directory relative to this library file's location
_MYSQL_SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
_MYSQL_ROOT_DIR="${_MYSQL_SCRIPT_DIR}/../.."

function _mysql_cleanup {
	docker kill "${MLRUN_DOCKER_CONTAINER_NAME}" 2>/dev/null || true
}

function _mysql_start_container {
	if [ -z "$MLRUN_MYSQL_IMAGE" ]; then
		echo "Environment variable MLRUN_MYSQL_IMAGE not set"
		exit 1
	fi
	docker run \
		--name="${MLRUN_DOCKER_CONTAINER_NAME}" \
		--rm \
		-p "${MLRUN_DOCKER_MYSQL_PORT}:3306" \
		-e MYSQL_ROOT_PASSWORD="${MLRUN_DOCKER_MYSQL_PASSWORD}" \
		-e MYSQL_ROOT_HOST="%" \
		-e MYSQL_DATABASE="${MLRUN_DOCKER_MYSQL_DATABASE}" \
		-d \
		"${MLRUN_MYSQL_IMAGE}" \
		--character-set-server=utf8 \
		--collation-server=utf8_bin
}

function _mysql_wait_ready {
	local times=0
	while ! docker exec "${MLRUN_DOCKER_CONTAINER_NAME}" \
			mysql --user=root --password="${MLRUN_DOCKER_MYSQL_PASSWORD}" \
			-e "status" > /dev/null 2>&1; do
		echo "Waiting for database connection..."
		sleep 2
		if [ $times -ge 60 ]; then
			echo "Timed out waiting for MySQL to become ready"
			exit 1
		fi
		times=$(( times + 1 ))
	done
}

function _mysql_setup_env {
	export MLRUN_HTTPDB__DSN="mysql+pymysql://root:${MLRUN_DOCKER_MYSQL_PASSWORD}@localhost:${MLRUN_DOCKER_MYSQL_PORT}/${MLRUN_DOCKER_MYSQL_DATABASE}"
	export PYTHONPATH="${_MYSQL_ROOT_DIR}:${_MYSQL_ROOT_DIR}/server/py"
}

function _mysql_full_setup {
	# Start container, wait for readiness, export env vars.
	# Caller must register the cleanup trap separately.
	_mysql_start_container
	_mysql_wait_ready
	_mysql_setup_env
}
