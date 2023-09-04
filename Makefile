# Copyright 2023 Iguazio
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

# THIS BLOCK IS FOR VARIABLES USER MAY OVERRIDE
MLRUN_VERSION ?= unstable
# pip requires the python version to be according to some regex (so "unstable" is not valid for example) this regex only
# allows us to have free text (like unstable) after the "+". on the contrary in a docker tag "+" is not a valid
# character so we're doing best effort - if the provided version doesn't look valid (like unstable), we prefix the
# version for the python package with 0.0.0+
# if the provided version includes a "+" we replace it with "-" for the docker tag
MLRUN_DOCKER_TAG ?= $(shell echo "$(MLRUN_VERSION)" | sed -E 's/\+/\-/g')
MLRUN_DOCKER_REPO ?= mlrun
# empty by default (dockerhub), can be set to something like "quay.io/".
# This will be used to tag the images built using this makefile
MLRUN_DOCKER_REGISTRY ?=
# empty by default (use cache), set it to anything to disable caching (will add flags to pip and docker commands to
# disable caching)
MLRUN_NO_CACHE ?=
MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX ?= ml-
# do not specify the patch version so that we can easily upgrade it when needed - it is determined by the base image
# mainly used for mlrun, base and mlrun-gpu. mlrun API version >= 1.3.0 should always have python 3.9
MLRUN_PYTHON_VERSION ?= 3.9
MLRUN_SKIP_COMPILE_SCHEMAS ?=
INCLUDE_PYTHON_VERSION_SUFFIX ?=
MLRUN_PIP_VERSION ?= 23.2.1
MLRUN_CACHE_DATE ?= $(shell date +%s)
# empty by default, can be set to something like "tag-name" which will cause to:
# 1. docker pull the same image with the given tag (cache image) before the build
# 2. add the --cache-from flag to the docker build
# 3. docker tag and push (also) the (updated) cache image
MLRUN_DOCKER_CACHE_FROM_TAG ?=
MLRUN_DOCKER_CACHE_FROM_REGISTRY ?= $(MLRUN_DOCKER_REGISTRY)
MLRUN_PUSH_DOCKER_CACHE_IMAGE ?=
MLRUN_GIT_ORG ?= mlrun
MLRUN_RELEASE_BRANCH ?= master
MLRUN_RAISE_ON_ERROR ?= true
MLRUN_SKIP_CLONE ?= false
MLRUN_RELEASE_NOTES_OUTPUT_FILE ?=
MLRUN_SYSTEM_TESTS_CLEAN_RESOURCES ?= true
MLRUN_GPU_CUDA_VERSION ?= 11.7.1-cudnn8-devel-ubuntu20.04

# THIS BLOCK IS FOR COMPUTED VARIABLES
MLRUN_DOCKER_IMAGE_PREFIX := $(if $(MLRUN_DOCKER_REGISTRY),$(strip $(MLRUN_DOCKER_REGISTRY))$(MLRUN_DOCKER_REPO),$(MLRUN_DOCKER_REPO))
MLRUN_CACHE_DOCKER_IMAGE_PREFIX := $(if $(MLRUN_DOCKER_CACHE_FROM_REGISTRY),$(strip $(MLRUN_DOCKER_CACHE_FROM_REGISTRY))$(MLRUN_DOCKER_REPO),$(MLRUN_DOCKER_REPO))
MLRUN_CORE_DOCKER_TAG_SUFFIX := -core
MLRUN_DOCKER_CACHE_FROM_FLAG :=
# if MLRUN_NO_CACHE passed we don't want to use cache, this is mainly used for cleaner if statements
MLRUN_USE_CACHE := $(if $(MLRUN_NO_CACHE),,true)
MLRUN_DOCKER_NO_CACHE_FLAG := $(if $(MLRUN_NO_CACHE),--no-cache,)
MLRUN_PIP_NO_CACHE_FLAG := $(if $(MLRUN_NO_CACHE),--no-cache-dir,)
# expected to be in the form of '-py<major><minor>' e.g. '-py39'
MLRUN_ANACONDA_PYTHON_DISTRIBUTION := $(shell echo "$(MLRUN_PYTHON_VERSION)" | awk -F. '{print "-py"$$1$$2}')
MLRUN_PYTHON_VERSION_SUFFIX := $(if $(INCLUDE_PYTHON_VERSION_SUFFIX),$(MLRUN_ANACONDA_PYTHON_DISTRIBUTION),)

MLRUN_OLD_VERSION_ESCAPED = $(shell echo "$(MLRUN_OLD_VERSION)" | sed 's/\./\\\./g')
MLRUN_BC_TESTS_OPENAPI_OUTPUT_PATH ?= $(shell pwd)

# if MLRUN_SYSTEM_TESTS_COMPONENT isn't set, we'll run all system tests
# if MLRUN_SYSTEM_TESTS_COMPONENT is set, we'll run only the system tests for the given component
# if MLRUN_SYSTEM_TESTS_COMPONENT starts with "no_", we'll ignore that component in the system tests
MLRUN_SYSTEM_TESTS_COMPONENT ?=
MLRUN_SYSTEM_TESTS_IGNORE_COMPONENT := $(shell echo "$(MLRUN_SYSTEM_TESTS_COMPONENT)" | sed 's/^no_\(.*\)/\1/g')
ifndef MLRUN_SYSTEM_TESTS_COMPONENT
	MLRUN_SYSTEM_TESTS_COMMAND_SUFFIX = "tests/system"
else ifeq ($(MLRUN_SYSTEM_TESTS_COMPONENT),$(MLRUN_SYSTEM_TESTS_IGNORE_COMPONENT))
	MLRUN_SYSTEM_TESTS_COMMAND_SUFFIX = "tests/system/$(MLRUN_SYSTEM_TESTS_COMPONENT)"
else
	MLRUN_SYSTEM_TESTS_COMMAND_SUFFIX = "--ignore=tests/system/$(MLRUN_SYSTEM_TESTS_COMPONENT) tests/system"
endif

.PHONY: help
help: ## Display available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: all
all:
	$(error please pick a target)

.PHONY: install-requirements
install-requirements: ## Install all requirements needed for development
	python -m pip install --upgrade $(MLRUN_PIP_NO_CACHE_FLAG) pip~=$(MLRUN_PIP_VERSION)
	python -m pip install \
		$(MLRUN_PIP_NO_CACHE_FLAG) \
		-r requirements.txt \
		-r extras-requirements.txt \
		-r dev-requirements.txt \
		-r dockerfiles/mlrun-api/requirements.txt \
		-r docs/requirements.txt

.PHONY: install-conda-requirements
install-conda-requirements: install-requirements ## Install all requirements needed for development with specific conda packages for arm64
	conda install --yes --file conda-arm64-requirements.txt

.PHONY: install-complete-requirements
install-complete-requirements: ## Install all requirements needed for development and testing
	python -m pip install --upgrade $(MLRUN_PIP_NO_CACHE_FLAG) pip~=$(MLRUN_PIP_VERSION)
	python -m pip install .[complete]

.PHONY: install-all-requirements
install-all-requirements: ## Install all requirements needed for development and testing
	python -m pip install --upgrade $(MLRUN_PIP_NO_CACHE_FLAG) pip~=$(MLRUN_PIP_VERSION)
	python -m pip install .[all]

.PHONY: create-migration-sqlite
create-migration-sqlite: ## Create a DB migration (MLRUN_MIGRATION_MESSAGE must be set)
	./automation/scripts/create_migration_sqlite.sh

.PHONY: create-migration-mysql
create-migration-mysql: ## Create a DB migration (MLRUN_MIGRATION_MESSAGE must be set)
	./automation/scripts/create_migration_mysql.sh

.PHONY: create-migration
create-migration: create-migration-sqlite create-migration-mysql
	@echo "Migrations created successfully"

.PHONY: bump-version
bump-version: ## Bump version in all needed places in code
ifndef MLRUN_NEW_VERSION
	$(error MLRUN_NEW_VERSION is undefined)
endif
ifndef MLRUN_OLD_VERSION
	$(error MLRUN_OLD_VERSION is undefined)
endif
	echo $(MLRUN_OLD_VERSION_ESCAPED)
	find . \( ! -regex '.*/\..*' \) -a \( -iname \*.md -o -iname \*.txt -o -iname \*.yaml -o -iname \*.yml \)  \
	-type f -print0 | xargs -0 sed -i '' -e 's/:$(MLRUN_OLD_VERSION_ESCAPED)/:$(MLRUN_NEW_VERSION)/g'
	find ./docs/install/*.yaml -type f -print0 | xargs -0 sed -i '' -e 's/{TAG:-.*}/{TAG:-$(MLRUN_NEW_VERSION)}/g'

.PHONY: update-version-file
update-version-file: ## Update the version file
	python ./automation/version/version_file.py ensure --mlrun-version $(MLRUN_VERSION)

.PHONY: build
build: docker-images package-wheel ## Build all artifacts
	@echo Done.

DEFAULT_DOCKER_IMAGES_RULES = \
	api \
	mlrun \
	mlrun-gpu \
	jupyter \
	base \
	log-collector

.PHONY: docker-images
docker-images: $(DEFAULT_DOCKER_IMAGES_RULES) ## Build all docker images
	@echo Done.

.PHONY: push-docker-images
push-docker-images: docker-images ## Push all docker images
	@echo "Pushing images concurrently $(DEFAULT_IMAGES)"
	@echo $(DEFAULT_IMAGES) | xargs -n 1 -P 5 docker push
	@echo Done.

.PHONY: print-docker-images
print-docker-images: ## Print all docker images
	@for image in $(DEFAULT_IMAGES); do \
		echo $$image ; \
	done


MLRUN_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/mlrun
MLRUN_CACHE_IMAGE_NAME := $(MLRUN_CACHE_DOCKER_IMAGE_PREFIX)/mlrun
MLRUN_IMAGE_NAME_TAGGED := $(MLRUN_IMAGE_NAME):$(MLRUN_DOCKER_TAG)$(MLRUN_PYTHON_VERSION_SUFFIX)
MLRUN_CACHE_IMAGE_NAME_TAGGED := $(MLRUN_CACHE_IMAGE_NAME):$(MLRUN_DOCKER_CACHE_FROM_TAG)$(MLRUN_PYTHON_VERSION_SUFFIX)
MLRUN_IMAGE_DOCKER_CACHE_FROM_FLAG := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_USE_CACHE)),--cache-from $(strip $(MLRUN_CACHE_IMAGE_NAME_TAGGED)),)
MLRUN_CACHE_IMAGE_PULL_COMMAND := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_USE_CACHE)), docker pull $(MLRUN_CACHE_IMAGE_NAME_TAGGED) || true,)
MLRUN_CACHE_IMAGE_PUSH_COMMAND := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_PUSH_DOCKER_CACHE_IMAGE)),docker tag $(MLRUN_IMAGE_NAME_TAGGED) $(MLRUN_CACHE_IMAGE_NAME_TAGGED) && docker push $(MLRUN_CACHE_IMAGE_NAME_TAGGED),)
DEFAULT_IMAGES += $(MLRUN_IMAGE_NAME_TAGGED)

.PHONY: mlrun
mlrun: update-version-file ## Build mlrun docker image
	$(MLRUN_CACHE_IMAGE_PULL_COMMAND)
	docker build \
		--file dockerfiles/mlrun/Dockerfile \
		--build-arg MLRUN_ANACONDA_PYTHON_DISTRIBUTION=$(MLRUN_ANACONDA_PYTHON_DISTRIBUTION) \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		--build-arg MLRUN_PIP_VERSION=$(MLRUN_PIP_VERSION) \
		$(MLRUN_IMAGE_DOCKER_CACHE_FROM_FLAG) \
		$(MLRUN_DOCKER_NO_CACHE_FLAG) \
		--tag $(MLRUN_IMAGE_NAME_TAGGED) .

.PHONY: push-mlrun
push-mlrun: mlrun ## Push mlrun docker image
	docker push $(MLRUN_IMAGE_NAME_TAGGED)
	$(MLRUN_CACHE_IMAGE_PUSH_COMMAND)

.PHONY: pull-mlrun
pull-mlrun: ## Pull mlrun docker image
	docker pull $(MLRUN_IMAGE_NAME_TAGGED)

MLRUN_GPU_PREBAKED_IMAGE_NAME_TAGGED := quay.io/mlrun/prebaked-cuda:$(MLRUN_GPU_CUDA_VERSION)
MLRUN_GPU_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/mlrun-gpu
MLRUN_GPU_CACHE_IMAGE_NAME := $(MLRUN_CACHE_DOCKER_IMAGE_PREFIX)/mlrun-gpu
MLRUN_GPU_IMAGE_NAME_TAGGED := $(MLRUN_GPU_IMAGE_NAME):$(MLRUN_DOCKER_TAG)$(MLRUN_PYTHON_VERSION_SUFFIX)
MLRUN_GPU_CACHE_IMAGE_NAME_TAGGED := $(MLRUN_GPU_CACHE_IMAGE_NAME):$(MLRUN_DOCKER_CACHE_FROM_TAG)$(MLRUN_PYTHON_VERSION_SUFFIX)
MLRUN_GPU_IMAGE_DOCKER_CACHE_FROM_FLAG := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_USE_CACHE)),--cache-from $(strip $(MLRUN_CACHE_IMAGE_NAME_TAGGED)),)
MLRUN_GPU_CACHE_IMAGE_PULL_COMMAND := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_USE_CACHE)), docker pull $(MLRUN_CACHE_IMAGE_NAME_TAGGED) || true,)
MLRUN_GPU_CACHE_IMAGE_PUSH_COMMAND := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_PUSH_DOCKER_CACHE_IMAGE)),docker tag $(MLRUN_GPU_IMAGE_NAME_TAGGED) $(MLRUN_GPU_CACHE_IMAGE_NAME_TAGGED) && docker push $(MLRUN_GPU_CACHE_IMAGE_NAME_TAGGED),)
DEFAULT_IMAGES += $(MLRUN_GPU_IMAGE_NAME_TAGGED)

.PHONY: mlrun-gpu
mlrun-gpu: update-version-file ## Build mlrun gpu docker image
	$(MLRUN_CACHE_IMAGE_PULL_COMMAND)
	docker build \
		--file dockerfiles/gpu/Dockerfile \
		--build-arg MLRUN_GPU_BASE_IMAGE=$(MLRUN_GPU_PREBAKED_IMAGE_NAME_TAGGED) \
		$(MLRUN_GPU_IMAGE_DOCKER_CACHE_FROM_FLAG) \
		$(MLRUN_DOCKER_NO_CACHE_FLAG) \
		--tag $(MLRUN_GPU_IMAGE_NAME_TAGGED) \
		.

.PHONY: push-mlrun-gpu
push-mlrun-gpu: mlrun-gpu ## Push mlrun gpu docker image
	docker push $(MLRUN_GPU_IMAGE_NAME_TAGGED)
	$(MLRUN_GPU_CACHE_IMAGE_PUSH_COMMAND)

.PHONY: pull-mlrun-gpu
pull-mlrun-gpu: ## Pull mlrun gpu docker image
	docker pull $(MLRUN_GPU_IMAGE_NAME_TAGGED)

.PHONY: prebake-mlrun-gpu
prebake-mlrun-gpu: ## Build prebake mlrun GPU based docker image
	docker build \
		--file dockerfiles/gpu/prebaked.Dockerfile \
		--build-arg CUDA_VER=$(MLRUN_GPU_CUDA_VERSION) \
		--build-arg MLRUN_ANACONDA_PYTHON_DISTRIBUTION=$(MLRUN_ANACONDA_PYTHON_DISTRIBUTION) \
		--tag $(MLRUN_GPU_PREBAKED_IMAGE_NAME_TAGGED) \
		.

.PHONY: push-prebake-mlrun-gpu
push-prebake-mlrun-gpu: ## Push prebake mlrun GPU based docker image
	docker push $(MLRUN_GPU_PREBAKED_IMAGE_NAME_TAGGED)

MLRUN_BASE_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)base
MLRUN_BASE_CACHE_IMAGE_NAME := $(MLRUN_CACHE_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)base
MLRUN_BASE_IMAGE_NAME_TAGGED := $(MLRUN_BASE_IMAGE_NAME):$(MLRUN_DOCKER_TAG)$(MLRUN_PYTHON_VERSION_SUFFIX)
MLRUN_CORE_BASE_IMAGE_NAME_TAGGED := $(MLRUN_BASE_IMAGE_NAME_TAGGED)$(MLRUN_CORE_DOCKER_TAG_SUFFIX)
MLRUN_BASE_CACHE_IMAGE_NAME_TAGGED := $(MLRUN_BASE_CACHE_IMAGE_NAME):$(MLRUN_DOCKER_CACHE_FROM_TAG)$(MLRUN_PYTHON_VERSION_SUFFIX)
MLRUN_BASE_IMAGE_DOCKER_CACHE_FROM_FLAG := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_USE_CACHE)),--cache-from $(strip $(MLRUN_BASE_CACHE_IMAGE_NAME_TAGGED)),)
MLRUN_BASE_CACHE_IMAGE_PUSH_COMMAND := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_PUSH_DOCKER_CACHE_IMAGE)),docker tag $(MLRUN_BASE_IMAGE_NAME_TAGGED) $(MLRUN_BASE_CACHE_IMAGE_NAME_TAGGED) && docker push $(MLRUN_BASE_CACHE_IMAGE_NAME_TAGGED),)
DEFAULT_IMAGES += $(MLRUN_BASE_IMAGE_NAME_TAGGED)

.PHONY: base-core
base-core: pull-cache update-version-file ## Build base core docker image
	docker build \
		--file dockerfiles/base/Dockerfile \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		--build-arg MLRUN_ANACONDA_PYTHON_DISTRIBUTION=$(MLRUN_ANACONDA_PYTHON_DISTRIBUTION) \
		--build-arg MLRUN_PIP_VERSION=$(MLRUN_PIP_VERSION) \
		$(MLRUN_DOCKER_CACHE_FROM_FLAG) \
		$(MLRUN_DOCKER_NO_CACHE_FLAG) \
		--tag $(MLRUN_CORE_BASE_IMAGE_NAME_TAGGED) .

.PHONY: base
base: base-core ## Build base docker image
	docker build \
		--file dockerfiles/common/Dockerfile \
		--build-arg MLRUN_BASE_IMAGE=$(MLRUN_CORE_BASE_IMAGE_NAME_TAGGED) \
		$(MLRUN_DOCKER_CACHE_FROM_FLAG) \
		$(MLRUN_DOCKER_NO_CACHE_FLAG) \
		--tag $(MLRUN_BASE_IMAGE_NAME_TAGGED) .

.PHONY: push-base
push-base: base ## Push base docker image
	docker push $(MLRUN_BASE_IMAGE_NAME_TAGGED)
	$(MLRUN_BASE_CACHE_IMAGE_PUSH_COMMAND)

.PHONY: pull-base
pull-base: ## Pull base docker image
	docker pull $(MLRUN_BASE_IMAGE_NAME_TAGGED)

MLRUN_JUPYTER_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/jupyter:$(MLRUN_DOCKER_TAG)$(MLRUN_PYTHON_VERSION_SUFFIX)
DEFAULT_IMAGES += $(MLRUN_JUPYTER_IMAGE_NAME)

.PHONY: jupyter
jupyter: update-version-file ## Build mlrun jupyter docker image
	docker build \
		--file dockerfiles/jupyter/Dockerfile \
		--build-arg MLRUN_PIP_VERSION=$(MLRUN_PIP_VERSION) \
		--build-arg MLRUN_CACHE_DATE=$(MLRUN_CACHE_DATE) \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		$(MLRUN_DOCKER_NO_CACHE_FLAG) \
		--tag $(MLRUN_JUPYTER_IMAGE_NAME) .

.PHONY: push-jupyter
push-jupyter: jupyter ## Push mlrun jupyter docker image
	docker push $(MLRUN_JUPYTER_IMAGE_NAME)

.PHONY: pull-jupyter
pull-jupyter: ## Pull mlrun jupyter docker image
	docker pull $(MLRUN_JUPYTER_IMAGE_NAME)

.PHONY: log-collector
log-collector: update-version-file
	@MLRUN_VERSION=$(MLRUN_VERSION) \
		MLRUN_DOCKER_REGISTRY=$(MLRUN_DOCKER_REGISTRY) \
		MLRUN_DOCKER_REPO=$(MLRUN_DOCKER_REPO) \
		MLRUN_DOCKER_TAG=$(MLRUN_DOCKER_TAG) \
		MLRUN_DOCKER_IMAGE_PREFIX=$(MLRUN_DOCKER_IMAGE_PREFIX) \
		make --no-print-directory -C $(shell pwd)/go log-collector

.PHONY: push-log-collector
push-log-collector: log-collector
	@MLRUN_VERSION=$(MLRUN_VERSION) \
		MLRUN_DOCKER_REGISTRY=$(MLRUN_DOCKER_REGISTRY) \
		MLRUN_DOCKER_REPO=$(MLRUN_DOCKER_REPO) \
		MLRUN_DOCKER_TAG=$(MLRUN_DOCKER_TAG) \
		MLRUN_DOCKER_IMAGE_PREFIX=$(MLRUN_DOCKER_IMAGE_PREFIX) \
		make --no-print-directory -C $(shell pwd)/go push-log-collector

.PHONY: pull-log-collector
pull-log-collector:
	@MLRUN_VERSION=$(MLRUN_VERSION) \
		MLRUN_DOCKER_REGISTRY=$(MLRUN_DOCKER_REGISTRY) \
		MLRUN_DOCKER_REPO=$(MLRUN_DOCKER_REPO) \
		MLRUN_DOCKER_TAG=$(MLRUN_DOCKER_TAG) \
		MLRUN_DOCKER_IMAGE_PREFIX=$(MLRUN_DOCKER_IMAGE_PREFIX) \
		make --no-print-directory -C $(shell pwd)/go pull-log-collector


.PHONY: compile-schemas
compile-schemas: ## Compile schemas
ifdef MLRUN_SKIP_COMPILE_SCHEMAS
	@echo "Skipping compile schemas"
else
	cd go && \
	  make compile-schemas
endif

MLRUN_API_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/mlrun-api
MLRUN_API_CACHE_IMAGE_NAME := $(MLRUN_CACHE_DOCKER_IMAGE_PREFIX)/mlrun-api
MLRUN_API_IMAGE_NAME_TAGGED := $(MLRUN_API_IMAGE_NAME):$(MLRUN_DOCKER_TAG)$(MLRUN_PYTHON_VERSION_SUFFIX)
MLRUN_API_CACHE_IMAGE_NAME_TAGGED := $(MLRUN_API_CACHE_IMAGE_NAME):$(MLRUN_DOCKER_CACHE_FROM_TAG)$(MLRUN_PYTHON_VERSION_SUFFIX)
MLRUN_API_IMAGE_DOCKER_CACHE_FROM_FLAG := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_USE_CACHE)),--cache-from $(strip $(MLRUN_API_CACHE_IMAGE_NAME_TAGGED)),)
MLRUN_API_CACHE_IMAGE_PULL_COMMAND := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_USE_CACHE)),docker pull $(MLRUN_API_CACHE_IMAGE_NAME_TAGGED) || true,)
MLRUN_API_CACHE_IMAGE_PUSH_COMMAND := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_PUSH_DOCKER_CACHE_IMAGE)),docker tag $(MLRUN_API_IMAGE_NAME_TAGGED) $(MLRUN_API_CACHE_IMAGE_NAME_TAGGED) && docker push $(MLRUN_API_CACHE_IMAGE_NAME_TAGGED),)
DEFAULT_IMAGES += $(MLRUN_API_IMAGE_NAME_TAGGED)

.PHONY: api
api: compile-schemas update-version-file ## Build mlrun-api docker image
	$(MLRUN_API_CACHE_IMAGE_PULL_COMMAND)
	docker build \
		--file dockerfiles/mlrun-api/Dockerfile \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		--build-arg MLRUN_PIP_VERSION=$(MLRUN_PIP_VERSION) \
		$(MLRUN_API_IMAGE_DOCKER_CACHE_FROM_FLAG) \
		$(MLRUN_DOCKER_NO_CACHE_FLAG) \
		--tag $(MLRUN_API_IMAGE_NAME_TAGGED) .

.PHONY: push-api
push-api: api ## Push api docker image
	docker push $(MLRUN_API_IMAGE_NAME_TAGGED)
	$(MLRUN_API_CACHE_IMAGE_PUSH_COMMAND)

.PHONY: pull-api
pull-api: ## Pull api docker image
	docker pull $(MLRUN_API_IMAGE_NAME_TAGGED)

MLRUN_TEST_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/test
MLRUN_TEST_CACHE_IMAGE_NAME := $(MLRUN_CACHE_DOCKER_IMAGE_PREFIX)/test
MLRUN_TEST_IMAGE_NAME_TAGGED := $(MLRUN_TEST_IMAGE_NAME):$(MLRUN_DOCKER_TAG)$(MLRUN_PYTHON_VERSION_SUFFIX)
MLRUN_TEST_CACHE_IMAGE_NAME_TAGGED := $(MLRUN_TEST_CACHE_IMAGE_NAME):$(MLRUN_DOCKER_CACHE_FROM_TAG)$(MLRUN_PYTHON_VERSION_SUFFIX)
MLRUN_TEST_IMAGE_DOCKER_CACHE_FROM_FLAG := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_USE_CACHE)),--cache-from $(strip $(MLRUN_TEST_CACHE_IMAGE_NAME_TAGGED)),)
MLRUN_TEST_CACHE_IMAGE_PULL_COMMAND := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_USE_CACHE)),docker pull $(MLRUN_TEST_CACHE_IMAGE_NAME_TAGGED) || true,)
MLRUN_TEST_CACHE_IMAGE_PUSH_COMMAND := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_PUSH_DOCKER_CACHE_IMAGE)),docker tag $(MLRUN_TEST_IMAGE_NAME_TAGGED) $(MLRUN_TEST_CACHE_IMAGE_NAME_TAGGED) && docker push $(MLRUN_TEST_CACHE_IMAGE_NAME_TAGGED),)

.PHONY: build-test
build-test: compile-schemas update-version-file ## Build test docker image
	$(MLRUN_TEST_CACHE_IMAGE_PULL_COMMAND)
	docker build \
		--file dockerfiles/test/Dockerfile \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		--build-arg MLRUN_PIP_VERSION=$(MLRUN_PIP_VERSION) \
		$(MLRUN_TEST_IMAGE_DOCKER_CACHE_FROM_FLAG) \
		$(MLRUN_DOCKER_NO_CACHE_FLAG) \
		--tag $(MLRUN_TEST_IMAGE_NAME_TAGGED) .

.PHONY: push-test
push-test: build-test ## Push test docker image
	docker push $(MLRUN_TEST_IMAGE_NAME_TAGGED)
	$(MLRUN_TEST_CACHE_IMAGE_PUSH_COMMAND)

MLRUN_SYSTEM_TEST_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/test-system:$(MLRUN_DOCKER_TAG)

.PHONY: build-test-system
build-test-system: compile-schemas update-version-file ## Build system tests docker image
	docker build \
		--file dockerfiles/test-system/Dockerfile \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		--build-arg MLRUN_PIP_VERSION=$(MLRUN_PIP_VERSION) \
		$(MLRUN_DOCKER_NO_CACHE_FLAG) \
		--tag $(MLRUN_SYSTEM_TEST_IMAGE_NAME) .

.PHONY: package-wheel
package-wheel: clean update-version-file ## Build python package wheel
	python setup.py bdist_wheel

.PHONY: publish-package
publish-package: package-wheel ## Publish python package wheel
	python -m twine upload dist/mlrun-*.whl

.PHONY: test-publish
test-publish: package-wheel ## Test python package publishing
	python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/mlrun-*.whl

.PHONY: clean
clean: ## Clean python package build artifacts
	rm -rf build
	rm -rf dist
	rm -rf mlrun.egg-info
	find . -name '*.pyc' -exec rm {} \;

.PHONY: test-dockerized
test-dockerized: build-test ## Run mlrun tests in docker container
	docker run \
		-t \
		--rm \
		--network='host' \
		-v /tmp:/tmp \
		-v /var/run/docker.sock:/var/run/docker.sock \
		$(MLRUN_TEST_IMAGE_NAME_TAGGED) make test

.PHONY: test
test: clean ## Run mlrun tests
	python -m pytest -v \
		--capture=no \
		--disable-warnings \
		--durations=100 \
		--ignore=tests/integration \
		--ignore=tests/system \
		--ignore=tests/rundb/test_httpdb.py \
		-rf \
		tests


.PHONY: test-integration-dockerized
test-integration-dockerized: build-test ## Run mlrun integration tests in docker container
	docker run \
		-t \
		--rm \
		--network='host' \
		-v /tmp:/tmp \
		-v /var/run/docker.sock:/var/run/docker.sock \
		$(MLRUN_TEST_IMAGE_NAME_TAGGED) make test-integration

.PHONY: test-integration
test-integration: clean ## Run mlrun integration tests
	python -m pytest -v \
		--capture=no \
		--disable-warnings \
		--durations=100 \
		-rf \
		tests/integration \
		tests/rundb/test_httpdb.py

.PHONY: test-migrations-dockerized
test-migrations-dockerized: build-test ## Run mlrun db migrations tests in docker container
	docker run \
		-t \
		--rm \
		--network='host' \
		-v /tmp:/tmp \
		-v /var/run/docker.sock:/var/run/docker.sock \
		$(MLRUN_TEST_IMAGE_NAME_TAGGED) make test-migrations

.PHONY: test-migrations
test-migrations: clean ## Run mlrun db migrations tests
	cd mlrun/api; \
	python -m pytest -v \
		--capture=no \
		--disable-warnings \
		--durations=100 \
		-rf \
		--test-alembic \
		migrations_sqlite/tests/*

.PHONY: test-system-dockerized
test-system-dockerized: build-test-system ## Run mlrun system tests in docker container
	docker run \
		--env MLRUN_SYSTEM_TESTS_CLEAN_RESOURCES=$(MLRUN_SYSTEM_TESTS_CLEAN_RESOURCES) \
		--env MLRUN_SYSTEM_TESTS_COMPONENT=$(MLRUN_SYSTEM_TESTS_COMPONENT) \
		--env MLRUN_VERSION=$(MLRUN_VERSION) \
		-t \
		--rm \
		$(MLRUN_SYSTEM_TEST_IMAGE_NAME)

.PHONY: test-system
test-system: ## Run mlrun system tests
	MLRUN_SYSTEM_TESTS_CLEAN_RESOURCES=$(MLRUN_SYSTEM_TESTS_CLEAN_RESOURCES) python -m pytest -v \
		--capture=no \
		--disable-warnings \
		--durations=100 \
		-rf \
		$(MLRUN_SYSTEM_TESTS_COMMAND_SUFFIX)

.PHONY: test-system-open-source
test-system-open-source: update-version-file ## Run mlrun system tests with opensource configuration
	MLRUN_SYSTEM_TESTS_CLEAN_RESOURCES=$(MLRUN_SYSTEM_TESTS_CLEAN_RESOURCES) python -m pytest -v \
		--capture=no \
		--disable-warnings \
		--durations=100 \
		-rf \
		-m "not enterprise" \
		$(MLRUN_SYSTEM_TESTS_COMMAND_SUFFIX)

.PHONY: test-package compile-schemas
test-package: ## Run mlrun package tests
	python ./automation/package_test/test.py run

.PHONY: test-go
test-go-unit: ## Run mlrun go unit tests
	cd go && \
		make test-unit-local

.PHONY: test-go-dockerized
test-go-unit-dockerized: ## Run mlrun go unit tests in docker container
	cd go && \
		make test-unit-dockerized

.PHONY: test-go
test-go-integration: ## Run mlrun go unit tests
	cd go && \
		make test-integration-local

.PHONY: test-go-dockerized
test-go-integration-dockerized: ## Run mlrun go integration tests in docker container
	cd go && \
		make test-integration-dockerized

.PHONY: run-api-undockerized
run-api-undockerized: ## Run mlrun api locally (un-dockerized)
	python -m mlrun db

.PHONY: run-api
run-api: api ## Run mlrun api (dockerized)
	docker run \
		--name mlrun-api \
		--detach \
		--publish 8080 \
		--add-host host.docker.internal:host-gateway \
		--env MLRUN_HTTPDB__DSN=$(MLRUN_HTTPDB__DSN) \
		--env MLRUN_LOG_LEVEL=$(MLRUN_LOG_LEVEL) \
		--env MLRUN_SECRET_STORES__TEST_MODE_MOCK_SECRETS=$(MLRUN_SECRET_STORES__TEST_MODE_MOCK_SECRETS) \
		$(MLRUN_API_IMAGE_NAME_TAGGED)

.PHONY: run-test-db
run-test-db:
	# clean up any previous test db container
	docker rm test-db --force || true
	docker run \
		--name=test-db \
		-v $(shell pwd):/mlrun \
		-p 3306:3306 \
		-e MYSQL_ROOT_PASSWORD="" \
		-e MYSQL_ALLOW_EMPTY_PASSWORD="true" \
		-e MYSQL_ROOT_HOST=% \
		-e MYSQL_DATABASE="mlrun" \
		-d \
		mysql/mysql-server:8.0 \
		--character-set-server=utf8 \
		--collation-server=utf8_bin \
		--sql_mode=""

.PHONY: html-docs
html-docs: ## Build html docs
	rm -f docs/external/*.md
	cd docs && make html

.PHONY: html-docs-dockerized
html-docs-dockerized: build-test ## Build html docs dockerized
	docker run \
		--rm \
		-v $(shell pwd)/docs/_build:/mlrun/docs/_build \
		$(MLRUN_TEST_IMAGE_NAME_TAGGED) \
		make html-docs

.PHONY: fmt
fmt: ## Format the code (using black and isort)
	@echo "Running black fmt..."
	python -m black .
	python -m isort .

.PHONY: lint-imports
lint-imports: ## Validates import dependencies
	@echo "Running import linter"
	lint-imports

.PHONY: lint
lint: flake8 fmt-check lint-imports ## Run lint on the code

.PHONY: fmt-check
fmt-check: ## Format and check the code (using black)
	@echo "Running black+isort fmt check..."
	python -m black --check --diff .
	python -m isort --check --diff .

.PHONY: flake8
flake8: ## Run flake8 lint
	@echo "Running flake8 lint..."
	python -m flake8 .

.PHONY: lint-go
lint-go:
	cd go && \
		make lint

.PHONY: fmt-go
fmt-go:
	cd go && \
		make fmt

.PHONY: release
release: ## Release a version
ifndef MLRUN_VERSION
	$(error MLRUN_VERSION is undefined)
endif
	TAG_SUFFIX=$$(echo $${MLRUN_VERSION%.*}.x); \
	BRANCH_NAME=$$(echo release/$$TAG_SUFFIX-latest); \
	git fetch origin $$BRANCH_NAME || EXIT_CODE=$$?; \
	echo $$EXIT_CODE; \
	if [ "$$EXIT_CODE" = "" ]; \
		then \
			echo "Branch $$BRANCH_NAME exists. Adding changes"; \
			git checkout $$BRANCH_NAME; \
			rm -rf /tmp/mlrun; \
			git clone --branch $(MLRUN_VERSION) https://github.com/$(MLRUN_GIT_ORG)/mlrun.git /tmp/mlrun; \
			find . -path ./.git -prune -o -exec rm -rf {} \; 2> /dev/null; \
			rsync -avr --exclude='.git/' /tmp/mlrun/ .; \
			git add -A; \
		else \
			echo "Creating new branch: $$BRANCH_NAME"; \
			git checkout --orphan $$BRANCH_NAME; \
	fi; \
	git commit -m "Adding $(MLRUN_VERSION) tag contents" --allow-empty; \
	git push origin $$BRANCH_NAME

.PHONY: test-backward-compatibility-dockerized
test-backward-compatibility-dockerized: build-test ## Run backward compatibility tests in docker container
ifndef MLRUN_BC_TESTS_BASE_CODE_PATH
	$(error MLRUN_BC_TESTS_BASE_CODE_PATH is undefined)
endif
	docker run \
	    -t \
	    --rm \
	    --network='host' \
	    -v /tmp:/tmp \
	    -v $(shell pwd):$(shell pwd) \
	    -v $(MLRUN_BC_TESTS_BASE_CODE_PATH):$(MLRUN_BC_TESTS_BASE_CODE_PATH) \
	    -v /var/run/docker.sock:/var/run/docker.sock \
	    --env MLRUN_BC_TESTS_BASE_CODE_PATH=$(MLRUN_BC_TESTS_BASE_CODE_PATH) \
	    --env MLRUN_BC_TESTS_OPENAPI_OUTPUT_PATH=$(shell pwd) \
	    --workdir=$(shell pwd) \
	    $(MLRUN_TEST_IMAGE_NAME_TAGGED) make test-backward-compatibility

.PHONY: test-backward-compatibility
test-backward-compatibility: ## Run backward compatibility tests
ifndef MLRUN_BC_TESTS_BASE_CODE_PATH
	$(error MLRUN_BC_TESTS_BASE_CODE_PATH is undefined)
endif
ifndef MLRUN_BC_TESTS_OPENAPI_OUTPUT_PATH
	$(error MLRUN_BC_TESTS_OPENAPI_OUTPUT_PATH is undefined)
endif
	export MLRUN_OPENAPI_JSON_NAME=mlrun_bc_base_oai.json && \
	python -m pytest -v --capture=no --disable-warnings --durations=100 $(MLRUN_BC_TESTS_BASE_CODE_PATH)/tests/api/api/test_docs.py::test_save_openapi_json && \
	export MLRUN_OPENAPI_JSON_NAME=mlrun_bc_head_oai.json && \
	python -m pytest -v --capture=no --disable-warnings --durations=100 tests/api/api/test_docs.py::test_save_openapi_json && \
	docker run --rm -t -v $(MLRUN_BC_TESTS_OPENAPI_OUTPUT_PATH):/specs:ro openapitools/openapi-diff:2.0.1 /specs/mlrun_bc_base_oai.json /specs/mlrun_bc_head_oai.json --fail-on-incompatible


.PHONY: release-notes
release-notes: ## Create release notes
ifndef MLRUN_VERSION
	$(error MLRUN_VERSION is undefined)
endif
ifndef MLRUN_OLD_VERSION
	$(error MLRUN_OLD_VERSION is undefined)
endif
ifndef MLRUN_RELEASE_BRANCH
	$(error MLRUN_RELEASE_BRANCH is undefined)
endif
	python ./automation/release_notes/generate.py run $(MLRUN_VERSION) $(MLRUN_OLD_VERSION) $(MLRUN_RELEASE_BRANCH) $(MLRUN_RAISE_ON_ERROR) $(MLRUN_RELEASE_NOTES_OUTPUT_FILE) $(MLRUN_SKIP_CLONE)


.PHONY: pull-cache
pull-cache: ## Pull images to be used as cache for build
ifdef MLRUN_DOCKER_CACHE_FROM_TAG
	targets="$(MAKECMDGOALS)" ; \
	for target in $$targets; do \
		image_name=$${target#"push-"} ; \
		tag=$(MLRUN_DOCKER_CACHE_FROM_TAG)$(MLRUN_PYTHON_VERSION_SUFFIX) ; \
		case "$$image_name" in \
		*models*) image_name=$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)$$image_name ;; \
		*base*) image_name=$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)$$image_name ;; \
		esac; \
		docker pull $(MLRUN_CACHE_DOCKER_IMAGE_PREFIX)/$$image_name:$$tag || true ; \
	done;
    ifneq (,$(findstring models,$(MAKECMDGOALS)))
        MLRUN_DOCKER_CACHE_FROM_FLAG := $(MLRUN_MODELS_IMAGE_DOCKER_CACHE_FROM_FLAG)
    else
        MLRUN_DOCKER_CACHE_FROM_FLAG := $(MLRUN_BASE_IMAGE_DOCKER_CACHE_FROM_FLAG)
    endif
endif
