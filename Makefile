# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

.PHONY: all
all:
	$(error please pick a target)

.PHONY: upload
upload: wheel
	python -m twine upload dist/mlrun-*.whl

.PHONY: wheel
wheel: clean
	python setup.py bdist_wheel

.PHONY: clean
clean:
	rm -rf build
	rm -rf dist
	rm -rf mlrun.egg-info
	find . -name '*.pyc' -exec rm {} \;


.PHONY: test
test: clean
	python -m pytest -v \
	    --disable-warnings \
	    -rf \
	    tests

.PHONY: run-httpd
run-httpd:
	python -m mlrun db

.PHONY: docker-db
docker-db:
	docker build \
	    -f Dockerfile.db \
	    --tag mlrun/mlrun-db \
	    .

.PHONY: docker-db-gunicorn
docker-db-gunicorn:
	docker build \
	    -f Dockerfile.db-gunicorn \
	    --tag mlrun/mlrun-db-gunicorn \
	    .
