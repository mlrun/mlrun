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

# THIS BLOCK IS FOR VARIABLES USER MAY OVERRIDE
MLRUN_VERSION ?= unstable
# pip requires the python version to be according to some regex (so "unstable" is not valid for example) this regex only
# allows us to have free text (like unstable) after the "+". on the contrary in a docker tag "+" is not a valid
# character so we're doing best effort - if the provided version doesn't look valid (like unstable), we prefix the
# version for the python package with 0.0.0+
# if the provided version includes a "+" we replace it with "-" for the docker tag
MLRUN_DOCKER_TAG ?= $(shell echo "$(MLRUN_VERSION)" | sed -E 's/\+/\-/g')
# if the provided version is a semver and followed by a "-" we replace its first occurrence with "+" to align with PEP 404
ifneq ($(shell echo "$(MLRUN_VERSION)" | grep -E "^[0-9]+\.[0-9]+\.[0-9]+-" | grep -vE "^[0-9]+\.[0-9]+\.[0-9]+-(a|b|rc)[0-9]+$$"),)
	MLRUN_PYTHON_PACKAGE_VERSION ?= $(shell echo "$(MLRUN_VERSION)" | sed "s/\-/\+/")
endif
ifeq ($(shell echo "$(MLRUN_VERSION)" | grep -E "^[0-9]+\.[0-9]+\.[0-9]+.*$$"),) # empty result from egrep
	MLRUN_PYTHON_PACKAGE_VERSION ?= 0.0.0+$(MLRUN_VERSION)
endif
MLRUN_PYTHON_PACKAGE_VERSION ?= $(MLRUN_VERSION)
MLRUN_DOCKER_REPO ?= mlrun
# empty by default (dockerhub), can be set to something like "quay.io/".
# This will be used to tag the images built using this makefile
MLRUN_DOCKER_REGISTRY ?=
# empty by default (use cache), set it to anything to disable caching (will add flags to pip and docker commands to
# disable caching)
MLRUN_NO_CACHE ?=
MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX ?= ml-
MLRUN_PYTHON_VERSION ?= 3.9.13
INCLUDE_PYTHON_VERSION_SUFFIX ?=
MLRUN_ANACONDA_PYTHON_DISTRIBUTION =$(shell echo "$(MLRUN_PYTHON_VERSION)" | awk -F. '{print "-py"$$1"."$$2}')
MLRUN_PYTHON_VERSION_SUFFIX = $(if $(INCLUDE_PYTHON_VERSION_SUFFIX),$(MLRUN_ANACONDA_PYTHON_DISTRIBUTION),)
MLRUN_PIP_VERSION ?= 22.3.0
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
MLRUN_SYSTEM_TESTS_CLEAN_RESOURCES ?= true
MLRUN_CUDA_VERSION = 11.7.0
MLRUN_TENSORFLOW_VERSION = 2.9.0
MLRUN_HOROVOD_VERSION = 0.25.0

# THIS BLOCK IS FOR COMPUTED VARIABLES
MLRUN_DOCKER_IMAGE_PREFIX := $(if $(MLRUN_DOCKER_REGISTRY),$(strip $(MLRUN_DOCKER_REGISTRY))$(MLRUN_DOCKER_REPO),$(MLRUN_DOCKER_REPO))
MLRUN_CACHE_DOCKER_IMAGE_PREFIX := $(if $(MLRUN_DOCKER_CACHE_FROM_REGISTRY),$(strip $(MLRUN_DOCKER_CACHE_FROM_REGISTRY))$(MLRUN_DOCKER_REPO),$(MLRUN_DOCKER_REPO))
MLRUN_CORE_DOCKER_TAG_SUFFIX := -core
MLRUN_DOCKER_CACHE_FROM_FLAG :=
# if MLRUN_NO_CACHE passed we don't want to use cache, this is mainly used for cleaner if statements
MLRUN_USE_CACHE := $(if $(MLRUN_NO_CACHE),,true)
MLRUN_DOCKER_NO_CACHE_FLAG := $(if $(MLRUN_NO_CACHE),--no-cache,)
MLRUN_PIP_NO_CACHE_FLAG := $(if $(MLRUN_NO_CACHE),--no-cache-dir,)

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

.PHONY: install-complete-requirements
install-complete-requirements: ## Install all requirements needed for development and testing
	python -m pip install --upgrade $(MLRUN_PIP_NO_CACHE_FLAG) pip~=$(MLRUN_PIP_VERSION)
	python -m pip install .[complete]

.PHONY: install-all-requirements
install-all-requirements: ## Install all requirements needed for development and testing
	python -m pip install --upgrade $(MLRUN_PIP_NO_CACHE_FLAG) pip~=$(MLRUN_PIP_VERSION)
	python -m pip install .[all]

.PHONY: create-migration-sqlite
create-migration-sqlite: export MLRUN_HTTPDB__DSN="sqlite:///$(shell pwd)/mlrun/api/migrations_sqlite/mlrun.db?check_same_thread=false"
create-migration-sqlite: ## Create a DB migration (MLRUN_MIGRATION_MESSAGE must be set)
ifndef MLRUN_MIGRATION_MESSAGE
	$(error MLRUN_MIGRATION_MESSAGE is undefined)
endif
	alembic -c ./mlrun/api/alembic.ini upgrade head
	alembic -c ./mlrun/api/alembic.ini revision --autogenerate -m "$(MLRUN_MIGRATION_MESSAGE)"

.PHONY: create-migration-mysql
create-migration-mysql: export MLRUN_HTTPDB__DSN="mysql+pymysql://root:pass@localhost:3306/mlrun"
create-migration-mysql: ## Create a DB migration (MLRUN_MIGRATION_MESSAGE must be set)
ifndef MLRUN_MIGRATION_MESSAGE
	$(error MLRUN_MIGRATION_MESSAGE is undefined)
endif
	docker run \
		--name=migration-db \
		--rm \
		-v $(shell pwd):/mlrun \
		-p 3306:3306 \
		-e MYSQL_ROOT_PASSWORD="pass" \
		-e MYSQL_ROOT_HOST=% \
		-e MYSQL_DATABASE="mlrun" \
		-d \
		mysql/mysql-server:8.0 \
		--character-set-server=utf8 \
		--collation-server=utf8_bin
	alembic -c ./mlrun/api/alembic_mysql.ini upgrade head
	alembic -c ./mlrun/api/alembic_mysql.ini revision --autogenerate -m "$(MLRUN_MIGRATION_MESSAGE)"
	docker kill migration-db
	docker rm migration-db

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
	python ./automation/version/version_file.py --mlrun-version $(MLRUN_PYTHON_PACKAGE_VERSION)

.PHONY: build
build: docker-images package-wheel ## Build all artifacts
	@echo Done.

DEFAULT_DOCKER_IMAGES_RULES = \
	api \
	mlrun \
	jupyter \
	base \
	models \
	models-gpu

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
MLRUN_IMAGE_NAME_TAGGED := $(MLRUN_IMAGE_NAME)$(MLRUN_PYTHON_VERSION_SUFFIX):$(MLRUN_DOCKER_TAG)
MLRUN_CACHE_IMAGE_NAME_TAGGED := $(MLRUN_CACHE_IMAGE_NAME)$(MLRUN_PYTHON_VERSION_SUFFIX):$(MLRUN_DOCKER_CACHE_FROM_TAG)
MLRUN_IMAGE_DOCKER_CACHE_FROM_FLAG := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_USE_CACHE)),--cache-from $(strip $(MLRUN_CACHE_IMAGE_NAME_TAGGED)),)
MLRUN_CACHE_IMAGE_PULL_COMMAND := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_USE_CACHE)), docker pull $(MLRUN_CACHE_IMAGE_NAME_TAGGED) || true,)
MLRUN_CACHE_IMAGE_PUSH_COMMAND := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_PUSH_DOCKER_CACHE_IMAGE)),docker tag $(MLRUN_IMAGE_NAME_TAGGED) $(MLRUN_CACHE_IMAGE_NAME_TAGGED) && docker push $(MLRUN_CACHE_IMAGE_NAME_TAGGED),)
DEFAULT_IMAGES += $(MLRUN_IMAGE_NAME_TAGGED)

.PHONY: mlrun
mlrun: update-version-file ## Build mlrun docker image
	$(MLRUN_CACHE_IMAGE_PULL_COMMAND)
	docker build \
		--file dockerfiles/mlrun/Dockerfile \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		--build-arg MLRUN_PIP_VERSION=$(MLRUN_PIP_VERSION) \
		$(MLRUN_IMAGE_DOCKER_CACHE_FROM_FLAG) \
		$(MLRUN_DOCKER_NO_CACHE_FLAG) \
		--tag $(MLRUN_IMAGE_NAME_TAGGED) .

.PHONY: push-mlrun
push-mlrun: mlrun ## Push mlrun docker image
	docker push $(MLRUN_IMAGE_NAME_TAGGED)
	$(MLRUN_CACHE_IMAGE_PUSH_COMMAND)


MLRUN_BASE_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)base
MLRUN_BASE_CACHE_IMAGE_NAME := $(MLRUN_CACHE_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)base
MLRUN_BASE_IMAGE_NAME_TAGGED := $(MLRUN_BASE_IMAGE_NAME)$(MLRUN_PYTHON_VERSION_SUFFIX):$(MLRUN_DOCKER_TAG)
MLRUN_CORE_BASE_IMAGE_NAME_TAGGED := $(MLRUN_BASE_IMAGE_NAME_TAGGED)$(MLRUN_CORE_DOCKER_TAG_SUFFIX)
MLRUN_BASE_CACHE_IMAGE_NAME_TAGGED := $(MLRUN_BASE_CACHE_IMAGE_NAME)$(MLRUN_PYTHON_VERSION_SUFFIX):$(MLRUN_DOCKER_CACHE_FROM_TAG)
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


MLRUN_MODELS_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)models
MLRUN_MODELS_CACHE_IMAGE_NAME := $(MLRUN_CACHE_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)models
MLRUN_MODELS_IMAGE_NAME_TAGGED := $(MLRUN_MODELS_IMAGE_NAME)$(MLRUN_PYTHON_VERSION_SUFFIX):$(MLRUN_DOCKER_TAG)
MLRUN_CORE_MODELS_IMAGE_NAME_TAGGED := $(MLRUN_MODELS_IMAGE_NAME_TAGGED)$(MLRUN_CORE_DOCKER_TAG_SUFFIX)
MLRUN_MODELS_CACHE_IMAGE_NAME_TAGGED := $(MLRUN_MODELS_CACHE_IMAGE_NAME)$(MLRUN_PYTHON_VERSION_SUFFIX):$(MLRUN_DOCKER_CACHE_FROM_TAG)
MLRUN_MODELS_IMAGE_DOCKER_CACHE_FROM_FLAG := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_USE_CACHE)),--cache-from $(strip $(MLRUN_MODELS_CACHE_IMAGE_NAME_TAGGED)),)
MLRUN_MODELS_CACHE_IMAGE_PUSH_COMMAND := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_PUSH_DOCKER_CACHE_IMAGE)),docker tag $(MLRUN_MODELS_IMAGE_NAME_TAGGED) $(MLRUN_MODELS_CACHE_IMAGE_NAME_TAGGED) && docker push $(MLRUN_MODELS_CACHE_IMAGE_NAME_TAGGED),)
DEFAULT_IMAGES += $(MLRUN_MODELS_IMAGE_NAME_TAGGED)

.PHONY: models-core
models-core: base-core ## Build models core docker image
	docker build \
		--file dockerfiles/models/Dockerfile \
		--build-arg MLRUN_BASE_IMAGE=$(MLRUN_CORE_BASE_IMAGE_NAME_TAGGED) \
		--build-arg TENSORFLOW_VERSION=$(MLRUN_TENSORFLOW_VERSION) \
		--build-arg HOROVOD_VERSION=$(MLRUN_HOROVOD_VERSION) \
		$(MLRUN_DOCKER_CACHE_FROM_FLAG) \
		$(MLRUN_DOCKER_NO_CACHE_FLAG) \
		--tag $(MLRUN_CORE_MODELS_IMAGE_NAME_TAGGED) .

.PHONY: models
models: models-core ## Build models docker image
	$(MLRUN_MODELS_CACHE_IMAGE_PULL_COMMAND)
	docker build \
		--file dockerfiles/common/Dockerfile \
		--build-arg MLRUN_BASE_IMAGE=$(MLRUN_CORE_MODELS_IMAGE_NAME_TAGGED) \
		$(MLRUN_DOCKER_CACHE_FROM_FLAG) \
		$(MLRUN_DOCKER_NO_CACHE_FLAG) \
		--tag $(MLRUN_MODELS_IMAGE_NAME_TAGGED) .

.PHONY: push-models
push-models: models ## Push models docker image
	docker push $(MLRUN_MODELS_IMAGE_NAME_TAGGED)
	$(MLRUN_MODELS_CACHE_IMAGE_PUSH_COMMAND)


MLRUN_MODELS_GPU_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)models-gpu
MLRUN_MODELS_GPU_CACHE_IMAGE_NAME := $(MLRUN_CACHE_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)models-gpu
MLRUN_MODELS_GPU_IMAGE_NAME_TAGGED := $(MLRUN_MODELS_GPU_IMAGE_NAME)$(MLRUN_PYTHON_VERSION_SUFFIX):$(MLRUN_DOCKER_TAG)
MLRUN_MODELS_GPU_CACHE_IMAGE_NAME_TAGGED := $(MLRUN_MODELS_GPU_CACHE_IMAGE_NAME)$(MLRUN_PYTHON_VERSION_SUFFIX):$(MLRUN_DOCKER_CACHE_FROM_TAG)
MLRUN_MODELS_GPU_IMAGE_DOCKER_CACHE_FROM_FLAG := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_USE_CACHE)),--cache-from $(strip $(MLRUN_MODELS_GPU_CACHE_IMAGE_NAME_TAGGED)),)
MLRUN_MODELS_GPU_CACHE_IMAGE_PULL_COMMAND := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_USE_CACHE)),docker pull $(MLRUN_MODELS_GPU_CACHE_IMAGE_NAME_TAGGED) || true,)
MLRUN_MODELS_GPU_CACHE_IMAGE_PUSH_COMMAND := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_PUSH_DOCKER_CACHE_IMAGE)),docker tag $(MLRUN_MODELS_GPU_IMAGE_NAME_TAGGED) $(MLRUN_MODELS_GPU_CACHE_IMAGE_NAME_TAGGED) && docker push $(MLRUN_MODELS_GPU_CACHE_IMAGE_NAME_TAGGED),)
DEFAULT_IMAGES += $(MLRUN_MODELS_GPU_IMAGE_NAME_TAGGED)

.PHONY: models-gpu
models-gpu: update-version-file ## Build models-gpu docker image
	$(MLRUN_MODELS_GPU_CACHE_IMAGE_PULL_COMMAND)
	docker build \
		--file dockerfiles/models-gpu/Dockerfile \
		--build-arg CUDA_VER=$(MLRUN_CUDA_VERSION) \
		$(MLRUN_MODELS_GPU_IMAGE_DOCKER_CACHE_FROM_FLAG) \
		$(MLRUN_DOCKER_NO_CACHE_FLAG) \
		--tag $(MLRUN_MODELS_GPU_IMAGE_NAME_TAGGED) \
		.

.PHONY: push-models-gpu
push-models-gpu: models-gpu ## Push models gpu docker image
	docker push $(MLRUN_MODELS_GPU_IMAGE_NAME_TAGGED)
	$(MLRUN_MODELS_GPU_CACHE_IMAGE_PUSH_COMMAND)

.PHONY: prebake-models-gpu
prebake-models-gpu: ## Build prebake models GPU docker image
	docker build \
		--file dockerfiles/models-gpu/prebaked.Dockerfile \
		--build-arg CUDA_VER=$(MLRUN_CUDA_VERSION) \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		--build-arg MLRUN_PIP_VERSION=$(MLRUN_PIP_VERSION) \
		--build-arg HOROVOD_VERSION=$(MLRUN_HOROVOD_VERSION) \
		--tag quay.io/mlrun/prebaked-cuda:$(MLRUN_CUDA_VERSION)-base-ubuntu20.04 \
		.

.PHONY: push-prebake-models-gpu
push-prebake-models-gpu: ## Push prebake models GPU docker image
	docker push quay.io/mlrun/prebaked-cuda:$(MLRUN_CUDA_VERSION)-base-ubuntu20.04


MLRUN_JUPYTER_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/jupyter$(MLRUN_PYTHON_VERSION_SUFFIX):$(MLRUN_DOCKER_TAG)
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


MLRUN_API_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/mlrun-api
MLRUN_API_CACHE_IMAGE_NAME := $(MLRUN_CACHE_DOCKER_IMAGE_PREFIX)/mlrun-api
MLRUN_API_IMAGE_NAME_TAGGED := $(MLRUN_API_IMAGE_NAME)$(MLRUN_PYTHON_VERSION_SUFFIX):$(MLRUN_DOCKER_TAG)
MLRUN_API_CACHE_IMAGE_NAME_TAGGED := $(MLRUN_API_CACHE_IMAGE_NAME)$(MLRUN_PYTHON_VERSION_SUFFIX):$(MLRUN_DOCKER_CACHE_FROM_TAG)
MLRUN_API_IMAGE_DOCKER_CACHE_FROM_FLAG := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_USE_CACHE)),--cache-from $(strip $(MLRUN_API_CACHE_IMAGE_NAME_TAGGED)),)
MLRUN_API_CACHE_IMAGE_PULL_COMMAND := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_USE_CACHE)),docker pull $(MLRUN_API_CACHE_IMAGE_NAME_TAGGED) || true,)
MLRUN_API_CACHE_IMAGE_PUSH_COMMAND := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_PUSH_DOCKER_CACHE_IMAGE)),docker tag $(MLRUN_API_IMAGE_NAME_TAGGED) $(MLRUN_API_CACHE_IMAGE_NAME_TAGGED) && docker push $(MLRUN_API_CACHE_IMAGE_NAME_TAGGED),)
DEFAULT_IMAGES += $(MLRUN_API_IMAGE_NAME_TAGGED)

.PHONY: api
api: update-version-file ## Build mlrun-api docker image
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


MLRUN_TEST_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/test
MLRUN_TEST_CACHE_IMAGE_NAME := $(MLRUN_CACHE_DOCKER_IMAGE_PREFIX)/test
MLRUN_TEST_IMAGE_NAME_TAGGED := $(MLRUN_TEST_IMAGE_NAME)$(MLRUN_PYTHON_VERSION_SUFFIX):$(MLRUN_DOCKER_TAG)
MLRUN_TEST_CACHE_IMAGE_NAME_TAGGED := $(MLRUN_TEST_CACHE_IMAGE_NAME)$(MLRUN_PYTHON_VERSION_SUFFIX):$(MLRUN_DOCKER_CACHE_FROM_TAG)
MLRUN_TEST_IMAGE_DOCKER_CACHE_FROM_FLAG := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_USE_CACHE)),--cache-from $(strip $(MLRUN_TEST_CACHE_IMAGE_NAME_TAGGED)),)
MLRUN_TEST_CACHE_IMAGE_PULL_COMMAND := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_USE_CACHE)),docker pull $(MLRUN_TEST_CACHE_IMAGE_NAME_TAGGED) || true,)
MLRUN_TEST_CACHE_IMAGE_PUSH_COMMAND := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_PUSH_DOCKER_CACHE_IMAGE)),docker tag $(MLRUN_TEST_IMAGE_NAME_TAGGED) $(MLRUN_TEST_CACHE_IMAGE_NAME_TAGGED) && docker push $(MLRUN_TEST_CACHE_IMAGE_NAME_TAGGED),)

.PHONY: build-test
build-test: update-version-file ## Build test docker image
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
build-test-system: update-version-file ## Build system tests docker image
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
		--ignore=tests/test_notebooks.py \
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
		tests/test_notebooks.py \
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

.PHONY: test-package
test-package: ## Run mlrun package tests
	python ./automation/package_test/test.py run

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
		$(MLRUN_API_IMAGE_NAME_TAGGED)

.PHONY: run-test-db
run-test-db:
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

.PHONY: lint
lint: flake8 fmt-check ## Run lint on the code

.PHONY: fmt-check
fmt-check: ## Format and check the code (using black)
	@echo "Running black+isort fmt check..."
	python -m black --check --diff .
	python -m isort --check --diff .

.PHONY: flake8
flake8: ## Run flake8 lint
	@echo "Running flake8 lint..."
	python -m flake8 .

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
	python ./automation/release_notes/generate.py run $(MLRUN_VERSION) $(MLRUN_OLD_VERSION) $(MLRUN_RELEASE_BRANCH)


.PHONY: pull-cache
pull-cache: ## Pull images to be used as cache for build
ifdef MLRUN_DOCKER_CACHE_FROM_TAG
	targets="$(MAKECMDGOALS)" ; \
	for target in $$targets; do \
		image_name=$${target#"push-"} ; \
		tag=$(MLRUN_DOCKER_CACHE_FROM_TAG) ; \
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
