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

MLRUN_VERSION ?= unstable
MLRUN_DOCKER_TAG ?= $(MLRUN_VERSION)
MLRUN_DOCKER_REPO ?= mlrun
# empty by default (dockerhub), can be set to something like "quay.io/".
# This will be used to tag the images built using this makefile
MLRUN_DOCKER_REGISTRY ?=
MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX ?= ml-
MLRUN_PYTHON_VERSION ?= 3.7
MLRUN_LEGACY_ML_PYTHON_VERSION ?= 3.6
MLRUN_MLUTILS_GITHUB_TAG ?= development
MLRUN_CACHE_DATE ?= $(shell date +%s)
# empty by default, can be set to something like "tag-name" which will cause to:
# 1. docker pull the same image with the given tag (cache image) before the build
# 2. add the --cache-from falg to the docker build
# 3. docker tag and push (also) the (updated) cache image
MLRUN_DOCKER_CACHE_FROM_TAG ?=
MLRUN_GIT_ORG ?= mlrun
MLRUN_RELEASE_BRANCH ?= master


MLRUN_DOCKER_IMAGE_PREFIX := $(if $(MLRUN_DOCKER_REGISTRY),$(strip $(MLRUN_DOCKER_REGISTRY))$(MLRUN_DOCKER_REPO),$(MLRUN_DOCKER_REPO))
MLRUN_LEGACY_DOCKER_TAG_SUFFIX := -py$(subst .,,$(MLRUN_LEGACY_ML_PYTHON_VERSION))
MLRUN_LEGACY_DOCKERFILE_DIR_NAME := py$(subst .,,$(MLRUN_LEGACY_ML_PYTHON_VERSION))

MLRUN_OLD_VERSION_ESCAPED = $(shell echo "$(MLRUN_OLD_VERSION)" | sed 's/\./\\\./g')

.PHONY: help
help: ## Display available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: all
all:
	$(error please pick a target)

.PHONY: install-requirements
install-requirements: ## Install all requirements needed for development
	python -m pip install --upgrade pip~=20.2.0
	python -m pip install \
		-r requirements.txt \
		-r dev-requirements.txt \
		-r dockerfiles/mlrun-api/requirements.txt \
		-r docs/requirements.txt

.PHONY: create-migration
create-migration: export MLRUN_HTTPDB__DSN="sqlite:///$(PWD)/mlrun/api/migrations/mlrun.db?check_same_thread=false"
create-migration: ## Create a DB migration (MLRUN_MIGRATION_MESSAGE must be set)
ifndef MLRUN_MIGRATION_MESSAGE
	$(error MLRUN_MIGRATION_MESSAGE is undefined)
endif
	alembic -c ./mlrun/api/alembic.ini upgrade head
	alembic -c ./mlrun/api/alembic.ini revision --autogenerate -m "$(MLRUN_MIGRATION_MESSAGE)"

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

.PHONY: update-version-file
update-version-file: automation ## Update the version file
	python ./automation/version/version_file.py --mlrun-version $(MLRUN_VERSION)

.PHONY: build
build: docker-images package-wheel ## Build all artifacts
	@echo Done.

DEFAULT_DOCKER_IMAGES_RULES = \
	api \
	mlrun \
	jupyter \
	base \
	base-legacy \
	models \
	models-legacy \
	models-gpu \
	models-gpu-legacy

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


MLRUN_BASE_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)base
MLRUN_BASE_IMAGE_NAME_TAGGED := $(MLRUN_BASE_IMAGE_NAME):$(MLRUN_DOCKER_TAG)
MLRUN_BASE_CACHE_IMAGE_NAME_TAGGED := $(MLRUN_BASE_IMAGE_NAME):$(MLRUN_DOCKER_CACHE_FROM_TAG)
MLRUN_BASE_IMAGE_DOCKER_CACHE_FROM_FLAG := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),--cache-from $(strip $(MLRUN_BASE_CACHE_IMAGE_NAME_TAGGED)),)
MLRUN_BASE_CACHE_IMAGE_PULL_COMMAND := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),docker pull $(MLRUN_BASE_CACHE_IMAGE_NAME_TAGGED) || true,)
MLRUN_BASE_CACHE_IMAGE_PUSH_COMMAND := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),docker tag $(MLRUN_BASE_IMAGE_NAME_TAGGED) $(MLRUN_BASE_CACHE_IMAGE_NAME_TAGGED) && docker push $(MLRUN_BASE_CACHE_IMAGE_NAME_TAGGED),)
DEFAULT_IMAGES += $(MLRUN_BASE_IMAGE_NAME_TAGGED)

.PHONY: base
base: mlrun ## Build base docker image
	$(MLRUN_BASE_CACHE_IMAGE_PULL_COMMAND)
	docker build \
		--file dockerfiles/base/Dockerfile \
		--build-arg MLRUN_VERSION=$(MLRUN_VERSION) \
		--build-arg MLRUN_MLUTILS_GITHUB_TAG=$(MLRUN_MLUTILS_GITHUB_TAG) \
		--build-arg MLRUN_MLUTILS_CACHE_DATE=$(MLRUN_CACHE_DATE) \
		$(MLRUN_BASE_IMAGE_DOCKER_CACHE_FROM_FLAG) \
		--tag $(MLRUN_BASE_IMAGE_NAME_TAGGED) .

.PHONY: push-base
push-base: base ## Push base docker image
	docker push $(MLRUN_BASE_IMAGE_NAME_TAGGED)
	$(MLRUN_BASE_CACHE_IMAGE_PUSH_COMMAND)


MLRUN_LEGACY_BASE_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)base
MLRUN_LEGACY_BASE_IMAGE_NAME_TAGGED := $(MLRUN_LEGACY_BASE_IMAGE_NAME):$(MLRUN_DOCKER_TAG)$(MLRUN_LEGACY_DOCKER_TAG_SUFFIX)
MLRUN_LEGACY_BASE_CACHE_IMAGE_NAME_TAGGED := $(MLRUN_LEGACY_BASE_IMAGE_NAME):$(MLRUN_DOCKER_CACHE_FROM_TAG)$(MLRUN_LEGACY_DOCKER_TAG_SUFFIX)
MLRUN_LEGACY_BASE_IMAGE_DOCKER_CACHE_FROM_FLAG := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),--cache-from $(strip $(MLRUN_LEGACY_BASE_CACHE_IMAGE_NAME_TAGGED)),)
MLRUN_LEGACY_BASE_CACHE_IMAGE_PULL_COMMAND := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),docker pull $(MLRUN_LEGACY_BASE_CACHE_IMAGE_NAME_TAGGED) || true,)
MLRUN_LEGACY_BASE_CACHE_IMAGE_PUSH_COMMAND := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),docker tag $(MLRUN_LEGACY_BASE_IMAGE_NAME_TAGGED) $(MLRUN_LEGACY_BASE_CACHE_IMAGE_NAME_TAGGED) && docker push $(MLRUN_LEGACY_BASE_CACHE_IMAGE_NAME_TAGGED),)
DEFAULT_IMAGES += $(MLRUN_LEGACY_BASE_IMAGE_NAME_TAGGED)

.PHONY: base-legacy
base-legacy: mlrun-legacy ## Build base legacy docker image
	$(MLRUN_LEGACY_BASE_CACHE_IMAGE_PULL_COMMAND)
	docker build \
		--file dockerfiles/base/Dockerfile \
		--build-arg MLRUN_VERSION=$(MLRUN_VERSION) \
		--build-arg MLRUN_MLUTILS_GITHUB_TAG=$(MLRUN_MLUTILS_GITHUB_TAG) \
		--build-arg MLRUN_MLUTILS_CACHE_DATE=$(MLRUN_CACHE_DATE) \
		$(MLRUN_LEGACY_BASE_IMAGE_DOCKER_CACHE_FROM_FLAG) \
		--tag $(MLRUN_LEGACY_BASE_IMAGE_NAME_TAGGED) .

.PHONY: push-base-legacy
push-base-legacy: base-legacy ## Push base legacy docker image
	docker push $(MLRUN_LEGACY_BASE_IMAGE_NAME_TAGGED)
	$(MLRUN_LEGACY_BASE_CACHE_IMAGE_PUSH_COMMAND)


MLRUN_MODELS_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)models
MLRUN_MODELS_IMAGE_NAME_TAGGED := $(MLRUN_MODELS_IMAGE_NAME):$(MLRUN_DOCKER_TAG)
MLRUN_MODELS_CACHE_IMAGE_NAME_TAGGED := $(MLRUN_MODELS_IMAGE_NAME):$(MLRUN_DOCKER_CACHE_FROM_TAG)
MLRUN_MODELS_IMAGE_DOCKER_CACHE_FROM_FLAG := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),--cache-from $(strip $(MLRUN_MODELS_CACHE_IMAGE_NAME_TAGGED)),)
MLRUN_MODELS_CACHE_IMAGE_PULL_COMMAND := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),docker pull $(MLRUN_MODELS_CACHE_IMAGE_NAME_TAGGED) || true,)
MLRUN_MODELS_CACHE_IMAGE_PUSH_COMMAND := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),docker tag $(MLRUN_MODELS_IMAGE_NAME_TAGGED) $(MLRUN_MODELS_CACHE_IMAGE_NAME_TAGGED) && docker push $(MLRUN_MODELS_CACHE_IMAGE_NAME_TAGGED),)
DEFAULT_IMAGES += $(MLRUN_MODELS_IMAGE_NAME_TAGGED)

.PHONY: models
models: base ## Build models docker image
	$(MLRUN_MODELS_CACHE_IMAGE_PULL_COMMAND)
	docker build \
		--file dockerfiles/models/Dockerfile \
		--build-arg MLRUN_VERSION=$(MLRUN_VERSION) \
		$(MLRUN_MODELS_IMAGE_DOCKER_CACHE_FROM_FLAG) \
		--tag $(MLRUN_MODELS_IMAGE_NAME_TAGGED) .

.PHONY: push-models
push-models: models ## Push models docker image
	docker push $(MLRUN_MODELS_IMAGE_NAME_TAGGED)
	$(MLRUN_MODELS_CACHE_IMAGE_PUSH_COMMAND)


MLRUN_LEGACY_MODELS_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)models
MLRUN_LEGACY_MODELS_IMAGE_NAME_TAGGED := $(MLRUN_LEGACY_MODELS_IMAGE_NAME):$(MLRUN_DOCKER_TAG)$(MLRUN_LEGACY_DOCKER_TAG_SUFFIX)
MLRUN_LEGACY_MODELS_CACHE_IMAGE_NAME_TAGGED := $(MLRUN_LEGACY_MODELS_IMAGE_NAME):$(MLRUN_DOCKER_CACHE_FROM_TAG)$(MLRUN_LEGACY_DOCKER_TAG_SUFFIX)
MLRUN_LEGACY_MODELS_IMAGE_DOCKER_CACHE_FROM_FLAG := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),--cache-from $(strip $(MLRUN_LEGACY_MODELS_CACHE_IMAGE_NAME_TAGGED)),)
MLRUN_LEGACY_MODELS_CACHE_IMAGE_PULL_COMMAND := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),docker pull $(MLRUN_LEGACY_MODELS_CACHE_IMAGE_NAME_TAGGED) || true,)
MLRUN_LEGACY_MODELS_CACHE_IMAGE_PUSH_COMMAND := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),docker tag $(MLRUN_LEGACY_MODELS_IMAGE_NAME_TAGGED) $(MLRUN_LEGACY_MODELS_CACHE_IMAGE_NAME_TAGGED) && docker push $(MLRUN_LEGACY_MODELS_CACHE_IMAGE_NAME_TAGGED),)
DEFAULT_IMAGES += $(MLRUN_LEGACY_MODELS_IMAGE_NAME_TAGGED)

.PHONY: models-legacy
models-legacy: base-legacy ## Build models legacy docker image
	$(MLRUN_LEGACY_MODELS_CACHE_IMAGE_PULL_COMMAND)
	docker build \
		--file dockerfiles/models/$(MLRUN_LEGACY_DOCKERFILE_DIR_NAME)/Dockerfile \
		--build-arg MLRUN_VERSION=$(MLRUN_VERSION) \
		$(MLRUN_LEGACY_MODELS_IMAGE_DOCKER_CACHE_FROM_FLAG) \
		--tag $(MLRUN_LEGACY_MODELS_IMAGE_NAME_TAGGED) .

.PHONY: push-models-legacy
push-models-legacy: models-legacy ## Push models legacy docker image
	docker push $(MLRUN_LEGACY_MODELS_IMAGE_NAME_TAGGED)
	$(MLRUN_LEGACY_MODELS_CACHE_IMAGE_PUSH_COMMAND)


MLRUN_MODELS_GPU_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)models-gpu
MLRUN_MODELS_GPU_IMAGE_NAME_TAGGED := $(MLRUN_MODELS_GPU_IMAGE_NAME):$(MLRUN_DOCKER_TAG)
MLRUN_MODELS_GPU_CACHE_IMAGE_NAME_TAGGED := $(MLRUN_MODELS_GPU_IMAGE_NAME):$(MLRUN_DOCKER_CACHE_FROM_TAG)
MLRUN_MODELS_GPU_IMAGE_DOCKER_CACHE_FROM_FLAG := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),--cache-from $(strip $(MLRUN_MODELS_GPU_CACHE_IMAGE_NAME_TAGGED)),)
MLRUN_MODELS_GPU_CACHE_IMAGE_PULL_COMMAND := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),docker pull $(MLRUN_MODELS_GPU_CACHE_IMAGE_NAME_TAGGED) || true,)
MLRUN_MODELS_GPU_CACHE_IMAGE_PUSH_COMMAND := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),docker tag $(MLRUN_MODELS_GPU_IMAGE_NAME_TAGGED) $(MLRUN_MODELS_GPU_CACHE_IMAGE_NAME_TAGGED) && docker push $(MLRUN_MODELS_GPU_CACHE_IMAGE_NAME_TAGGED),)
DEFAULT_IMAGES += $(MLRUN_MODELS_GPU_IMAGE_NAME_TAGGED)

.PHONY: models-gpu
models-gpu: update-version-file ## Build models-gpu docker image
	$(MLRUN_MODELS_GPU_CACHE_IMAGE_PULL_COMMAND)
	docker build \
		--file dockerfiles/models-gpu/Dockerfile \
		--build-arg MLRUN_MLUTILS_GITHUB_TAG=$(MLRUN_MLUTILS_GITHUB_TAG) \
		--build-arg MLRUN_MLUTILS_CACHE_DATE=$(MLRUN_CACHE_DATE) \
		$(MLRUN_MODELS_GPU_IMAGE_DOCKER_CACHE_FROM_FLAG) \
		--tag $(MLRUN_MODELS_GPU_IMAGE_NAME_TAGGED) .

.PHONY: push-models-gpu
push-models-gpu: models-gpu ## Push models gpu docker image
	docker push $(MLRUN_MODELS_GPU_IMAGE_NAME_TAGGED)
	$(MLRUN_MODELS_GPU_CACHE_IMAGE_PUSH_COMMAND)


MLRUN_LEGACY_MODELS_GPU_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)models-gpu
MLRUN_LEGACY_MODELS_GPU_IMAGE_NAME_TAGGED := $(MLRUN_LEGACY_MODELS_GPU_IMAGE_NAME):$(MLRUN_DOCKER_TAG)$(MLRUN_LEGACY_DOCKER_TAG_SUFFIX)
MLRUN_LEGACY_MODELS_GPU_CACHE_IMAGE_NAME_TAGGED := $(MLRUN_LEGACY_MODELS_GPU_IMAGE_NAME):$(MLRUN_DOCKER_CACHE_FROM_TAG)$(MLRUN_LEGACY_DOCKER_TAG_SUFFIX)
MLRUN_LEGACY_MODELS_GPU_IMAGE_DOCKER_CACHE_FROM_FLAG := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),--cache-from $(strip $(MLRUN_LEGACY_MODELS_GPU_CACHE_IMAGE_NAME_TAGGED)),)
MLRUN_LEGACY_MODELS_GPU_CACHE_IMAGE_PULL_COMMAND := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),docker pull $(MLRUN_LEGACY_MODELS_GPU_CACHE_IMAGE_NAME_TAGGED) || true,)
MLRUN_LEGACY_MODELS_GPU_CACHE_IMAGE_PUSH_COMMAND := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),docker tag $(MLRUN_LEGACY_MODELS_GPU_IMAGE_NAME_TAGGED) $(MLRUN_LEGACY_MODELS_GPU_CACHE_IMAGE_NAME_TAGGED) && docker push $(MLRUN_LEGACY_MODELS_GPU_CACHE_IMAGE_NAME_TAGGED),)
DEFAULT_IMAGES += $(MLRUN_LEGACY_MODELS_GPU_IMAGE_NAME_TAGGED)

.PHONY: models-gpu-legacy
models-gpu-legacy: update-version-file ## Build models-gpu legacy docker image
	$(MLRUN_LEGACY_MODELS_GPU_CACHE_IMAGE_PULL_COMMAND)
	docker build \
		--file dockerfiles/models-gpu/$(MLRUN_LEGACY_DOCKERFILE_DIR_NAME)/Dockerfile \
		--build-arg MLRUN_MLUTILS_GITHUB_TAG=$(MLRUN_MLUTILS_GITHUB_TAG) \
		--build-arg MLRUN_MLUTILS_CACHE_DATE=$(MLRUN_CACHE_DATE) \
		$(MLRUN_LEGACY_MODELS_GPU_IMAGE_DOCKER_CACHE_FROM_FLAG) \
		--tag $(MLRUN_LEGACY_MODELS_GPU_IMAGE_NAME_TAGGED) .

.PHONY: push-models-gpu-legacy
push-models-gpu-legacy: models-gpu-legacy ## Push models gpu legacy docker image
	docker push $(MLRUN_LEGACY_MODELS_GPU_IMAGE_NAME_TAGGED)
	$(MLRUN_LEGACY_MODELS_GPU_CACHE_IMAGE_PUSH_COMMAND)


MLRUN_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/mlrun
MLRUN_IMAGE_NAME_TAGGED := $(MLRUN_IMAGE_NAME):$(MLRUN_DOCKER_TAG)
MLRUN_CACHE_IMAGE_NAME_TAGGED := $(MLRUN_IMAGE_NAME):$(MLRUN_DOCKER_CACHE_FROM_TAG)
MLRUN_IMAGE_DOCKER_CACHE_FROM_FLAG := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),--cache-from $(strip $(MLRUN_CACHE_IMAGE_NAME_TAGGED)),)
MLRUN_CACHE_IMAGE_PULL_COMMAND := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),docker pull $(MLRUN_CACHE_IMAGE_NAME_TAGGED) || true,)
MLRUN_CACHE_IMAGE_PUSH_COMMAND := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),docker tag $(MLRUN_IMAGE_NAME_TAGGED) $(MLRUN_CACHE_IMAGE_NAME_TAGGED) && docker push $(MLRUN_CACHE_IMAGE_NAME_TAGGED),)
DEFAULT_IMAGES += $(MLRUN_IMAGE_NAME_TAGGED)

.PHONY: mlrun
mlrun: update-version-file ## Build mlrun docker image
	$(MLRUN_CACHE_IMAGE_PULL_COMMAND)
	docker build \
		--file dockerfiles/mlrun/Dockerfile \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		$(MLRUN_IMAGE_DOCKER_CACHE_FROM_FLAG) \
		--tag $(MLRUN_IMAGE_NAME_TAGGED) .

.PHONY: push-mlrun
push-mlrun: mlrun ## Push mlrun docker image
	docker push $(MLRUN_IMAGE_NAME_TAGGED)
	$(MLRUN_CACHE_IMAGE_PUSH_COMMAND)


MLRUN_LEGACY_MLRUN_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/mlrun
MLRUN_LEGACY_MLRUN_IMAGE_NAME_TAGGED := $(MLRUN_LEGACY_MLRUN_IMAGE_NAME):$(MLRUN_DOCKER_TAG)$(MLRUN_LEGACY_DOCKER_TAG_SUFFIX)
MLRUN_LEGACY_MLRUN_CACHE_IMAGE_NAME_TAGGED := $(MLRUN_LEGACY_MLRUN_IMAGE_NAME):$(MLRUN_DOCKER_CACHE_FROM_TAG)$(MLRUN_LEGACY_DOCKER_TAG_SUFFIX)
MLRUN_LEGACY_MLRUN_IMAGE_DOCKER_CACHE_FROM_FLAG := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),--cache-from $(strip $(MLRUN_LEGACY_MLRUN_CACHE_IMAGE_NAME_TAGGED)),)
MLRUN_LEGACY_MLRUN_CACHE_IMAGE_PULL_COMMAND := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),docker pull $(MLRUN_LEGACY_MLRUN_CACHE_IMAGE_NAME_TAGGED) || true,)
MLRUN_LEGACY_MLRUN_CACHE_IMAGE_PUSH_COMMAND := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),docker tag $(MLRUN_LEGACY_MLRUN_IMAGE_NAME_TAGGED) $(MLRUN_LEGACY_MLRUN_CACHE_IMAGE_NAME_TAGGED) && docker push $(MLRUN_LEGACY_MLRUN_CACHE_IMAGE_NAME_TAGGED),)
DEFAULT_IMAGES += $(MLRUN_LEGACY_MLRUN_IMAGE_NAME_TAGGED)

.PHONY: mlrun-legacy
mlrun-legacy: update-version-file ## Build mlrun legacy docker image
	$(MLRUN_LEGACY_MLRUN_CACHE_IMAGE_PULL_COMMAND)
	docker build \
		--file dockerfiles/mlrun/Dockerfile \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_LEGACY_ML_PYTHON_VERSION) \
		$(MLRUN_LEGACY_MLRUN_IMAGE_DOCKER_CACHE_FROM_FLAG) \
		--tag $(MLRUN_LEGACY_MLRUN_IMAGE_NAME_TAGGED) .

.PHONY: push-mlrun-legacy
push-mlrun-legacy: mlrun-legacy ## Push mlrun legacy docker image
	docker push $(MLRUN_LEGACY_MLRUN_IMAGE_NAME_TAGGED)
	$(MLRUN_LEGACY_MLRUN_CACHE_IMAGE_PUSH_COMMAND)


MLRUN_JUPYTER_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/jupyter:$(MLRUN_DOCKER_TAG)
DEFAULT_IMAGES += $(MLRUN_JUPYTER_IMAGE_NAME)

.PHONY: jupyter
jupyter: update-version-file ## Build mlrun jupyter docker image
	docker build \
		--file dockerfiles/jupyter/Dockerfile \
		--build-arg MLRUN_CACHE_DATE=$(MLRUN_CACHE_DATE) \
		--tag $(MLRUN_JUPYTER_IMAGE_NAME) .

.PHONY: push-jupyter
push-jupyter: jupyter ## Push mlrun jupyter docker image
	docker push $(MLRUN_JUPYTER_IMAGE_NAME)


MLRUN_SERVING_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)serving:$(MLRUN_DOCKER_TAG)

.PHONY: serving
serving: update-version-file ## Build serving docker image
	docker build \
		--file dockerfiles/serving/Dockerfile \
		--build-arg MLRUN_DOCKER_TAG=$(MLRUN_DOCKER_TAG) \
		--build-arg MLRUN_DOCKER_REPO=$(MLRUN_DOCKER_REPO) \
		--build-arg MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX=$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX) \
		--tag $(MLRUN_SERVING_IMAGE_NAME) .

.PHONY: push-serving
push-serving: serving ## Push serving docker image
	docker push $(MLRUN_SERVING_IMAGE_NAME)


MLRUN_API_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/mlrun-api
MLRUN_API_IMAGE_NAME_TAGGED := $(MLRUN_API_IMAGE_NAME):$(MLRUN_DOCKER_TAG)
MLRUN_API_CACHE_IMAGE_NAME_TAGGED := $(MLRUN_API_IMAGE_NAME):$(MLRUN_DOCKER_CACHE_FROM_TAG)
MLRUN_API_IMAGE_DOCKER_CACHE_FROM_FLAG := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),--cache-from $(strip $(MLRUN_API_CACHE_IMAGE_NAME_TAGGED)),)
MLRUN_API_CACHE_IMAGE_PULL_COMMAND := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),docker pull $(MLRUN_API_CACHE_IMAGE_NAME_TAGGED) || true,)
MLRUN_API_CACHE_IMAGE_PUSH_COMMAND := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),docker tag $(MLRUN_API_IMAGE_NAME_TAGGED) $(MLRUN_API_CACHE_IMAGE_NAME_TAGGED) && docker push $(MLRUN_API_CACHE_IMAGE_NAME_TAGGED),)
DEFAULT_IMAGES += $(MLRUN_API_IMAGE_NAME_TAGGED)

.PHONY: api
api: update-version-file ## Build mlrun-api docker image
	$(MLRUN_API_CACHE_IMAGE_PULL_COMMAND)
	docker build \
		--file dockerfiles/mlrun-api/Dockerfile \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		$(MLRUN_API_IMAGE_DOCKER_CACHE_FROM_FLAG) \
		--tag $(MLRUN_API_IMAGE_NAME_TAGGED) .

.PHONY: push-api
push-api: api ## Push api docker image
	docker push $(MLRUN_API_IMAGE_NAME_TAGGED)
	$(MLRUN_API_CACHE_IMAGE_PUSH_COMMAND)


MLRUN_TEST_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/test
MLRUN_TEST_IMAGE_NAME_TAGGED := $(MLRUN_TEST_IMAGE_NAME):$(MLRUN_DOCKER_TAG)
MLRUN_TEST_CACHE_IMAGE_NAME_TAGGED := $(MLRUN_TEST_IMAGE_NAME):$(MLRUN_DOCKER_CACHE_FROM_TAG)
MLRUN_TEST_IMAGE_DOCKER_CACHE_FROM_FLAG := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),--cache-from $(strip $(MLRUN_TEST_CACHE_IMAGE_NAME_TAGGED)),)
MLRUN_TEST_CACHE_IMAGE_PULL_COMMAND := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),docker pull $(MLRUN_TEST_CACHE_IMAGE_NAME_TAGGED) || true,)
MLRUN_TEST_CACHE_IMAGE_PUSH_COMMAND := $(if $(MLRUN_DOCKER_CACHE_FROM_TAG),docker tag $(MLRUN_TEST_IMAGE_NAME_TAGGED) $(MLRUN_TEST_CACHE_IMAGE_NAME_TAGGED) && docker push $(MLRUN_TEST_CACHE_IMAGE_NAME_TAGGED),)

.PHONY: build-test
build-test: update-version-file ## Build test docker image
	$(MLRUN_TEST_CACHE_IMAGE_PULL_COMMAND)
	docker build \
		--file dockerfiles/test/Dockerfile \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		$(MLRUN_TEST_IMAGE_DOCKER_CACHE_FROM_FLAG) \
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
		--ignore=tests/integration \
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
		-rf \
		--test-alembic \
		migrations/tests/*

.PHONY: test-system-dockerized
test-system-dockerized: build-test-system ## Run mlrun system tests in docker container
	docker run -t --rm $(MLRUN_SYSTEM_TEST_IMAGE_NAME)

.PHONY: test-system
test-system: ## Run mlrun system tests
	python -m pytest -v \
		--capture=no \
		--disable-warnings \
		-rf \
		tests/system

.PHONY: test-system-open-source
test-system-open-source: ## Run mlrun system tests with opensource configuration
	python -m pytest -v \
		--capture=no \
		--disable-warnings \
		-rsf \
		-m "not enterprise" \
		tests/system

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
		$(MLRUN_API_IMAGE_NAME_TAGGED)

.PHONY: html-docs
html-docs: ## Build html docs
	rm -f docs/external/*.md
	cd docs && make html

.PHONY: html-docs-dockerized
html-docs-dockerized: build-test ## Build html docs dockerized
	docker run \
		--rm \
		-v $(PWD)/docs/_build:/mlrun/docs/_build \
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
