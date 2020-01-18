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

.PHONY: circleci
circleci:
	docker build -f Dockerfile.test -t mlrun/test .
	-docker network create mlrun
	docker run \
	    -v /var/run/docker.sock:/var/run/docker.sock \
	    --network mlrun \
	    mlrun/test make test

.PHONY: docs
docs:
	sphinx-apidoc \
	    --append-syspath \
	    --doc-author Iguazio \
	    --doc-version $(shell python setup.py --version) \
	    --extensions numpydoc \
	    --force \
	    --full \
	    --output docs \
	    --private \
	    mlrun
	# Fix sys.path in conf.py
	sed -i "s#'.*/mlrun')#'..')#" docs/conf.py
	# Add api to docs
	sed -i s'/   mlrun/   api\n   mlrun/' docs/index.rst
	# Special requirements file
	cp requirements.txt docs
	echo numpydoc >> docs/requirements.txt

.PHONY: html-docs
html-docs:
	cd docs && make html
