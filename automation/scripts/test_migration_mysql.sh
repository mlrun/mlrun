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

function cleanup {
	docker kill migration-db
}
trap cleanup SIGHUP SIGINT SIGTERM EXIT


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR="${SCRIPT_DIR}/../.."

export MLRUN_HTTPDB__DSN="mysql+pymysql://root:pass@localhost:3306/mlrun"

docker run \
	--name=migration-db \
	--rm \
	-p 3306:3306 \
	-e MYSQL_ROOT_PASSWORD="pass" \
	-e MYSQL_ROOT_HOST="%" \
	-e MYSQL_DATABASE="mlrun" \
	-d \
	gcr.io/iguazio/mlrun-mysql:8.0 \
	--character-set-server=utf8 \
	--collation-server=utf8_bin

times=0
while ! docker exec migration-db mysql --user=root --password=pass -e "status" > /dev/null 2>&1; do
	echo "Waiting for database connection..."
	sleep 2
	if [ $times -ge 60 ]; then
		exit 1
	fi
	times=$(( times + 1 ))
done

export PYTHONPATH=$ROOT_DIR

python -m pytest -v \
		--capture=no \
		--disable-warnings \
		--durations=100 \
		-rf \
		${ROOT_DIR}/server/api/migrations/tests/*
