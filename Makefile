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

MLRUN_DOCKER_TAG ?= latest
MLRUN_DOCKER_REPO ?= mlrun
MLRUN_DOCKER_REGISTRY ?=  # empty be default (dockerhub), can be set to something like "quay.io/"
MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX ?= ml-
MLRUN_PACKAGE_TAG ?= 69ab04c505d06e3a48bac68031d224d3043f6c9c
MLRUN_GITHUB_REPO ?= mlrun
MLRUN_PYTHON_VERSION ?= 3.7
MLRUN_LEGACY_ML_PYTHON_VERSION ?= 3.6
MLRUN_MLUTILS_GITHUB_TAG ?= 3794e129cebc4d0dfef8d22f303d9f33f30358b9


MLRUN_DOCKER_IMAGE_PREFIX := $(if $(MLRUN_DOCKER_REGISTRY),$(strip $(MLRUN_DOCKER_REGISTRY))$(MLRUN_DOCKER_REPO),$(MLRUN_DOCKER_REPO))
MLRUN_LEGACY_DOCKER_TAG_SUFFIX := -py$(subst .,,$(MLRUN_LEGACY_ML_PYTHON_VERSION))
MLRUN_LEGACY_DOCKERFILE_DIR_NAME := py$(subst .,,$(MLRUN_LEGACY_ML_PYTHON_VERSION))

.PHONY: help
help: ## Display available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: all
all:
	$(error please pick a target)

.PHONY: build
build: docker-images package-wheel ## Build all artifacts
	@echo Done.

DEFAULT_DOCKER_IMAGES_RULES = \
	api \
	mlrun \
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


MLRUN_BASE_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)base:$(MLRUN_DOCKER_TAG)
DEFAULT_IMAGES += $(MLRUN_BASE_IMAGE_NAME)

.PHONY: base
base: ## Build base docker image
	docker build \
		--file dockerfiles/base/Dockerfile \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		--build-arg MLRUN_PACKAGE_TAG=$(MLRUN_PACKAGE_TAG) \
		--build-arg MLRUN_MLUTILS_GITHUB_TAG=${MLRUN_MLUTILS_GITHUB_TAG} \
		--build-arg MLRUN_GITHUB_REPO=$(MLRUN_GITHUB_REPO) \
		--tag $(MLRUN_BASE_IMAGE_NAME) .

.PHONY: push-base
push-base: base ## Push base docker image
	docker push $(MLRUN_BASE_IMAGE_NAME)


MLRUN_LEGACY_BASE_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)base:$(MLRUN_DOCKER_TAG)$(MLRUN_LEGACY_DOCKER_TAG_SUFFIX)
DEFAULT_IMAGES += $(MLRUN_LEGACY_BASE_IMAGE_NAME)

.PHONY: base-legacy
base-legacy: ## Build base legacy docker image
	docker build \
		--file dockerfiles/base/$(MLRUN_LEGACY_DOCKERFILE_DIR_NAME)/Dockerfile \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_LEGACY_ML_PYTHON_VERSION) \
		--build-arg MLRUN_PACKAGE_TAG=$(MLRUN_PACKAGE_TAG) \
		--build-arg MLRUN_MLUTILS_GITHUB_TAG=$(MLRUN_MLUTILS_GITHUB_TAG) \
		--build-arg MLRUN_GITHUB_REPO=$(MLRUN_GITHUB_REPO) \
		--tag $(MLRUN_LEGACY_BASE_IMAGE_NAME) .

.PHONY: push-base-legacy
push-base-legacy: base-legacy ## Push base legacy docker image
	docker push $(MLRUN_LEGACY_BASE_IMAGE_NAME)


MLRUN_MODELS_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)models:$(MLRUN_DOCKER_TAG)
DEFAULT_IMAGES += $(MLRUN_MODELS_IMAGE_NAME)

.PHONY: models
models: ## Build models docker image
	docker build \
		--file dockerfiles/models/Dockerfile \
		--build-arg MLRUN_PACKAGE_TAG=$(MLRUN_PACKAGE_TAG) \
		--build-arg MLRUN_MLUTILS_GITHUB_TAG=$(MLRUN_MLUTILS_GITHUB_TAG) \
		--build-arg MLRUN_GITHUB_REPO=$(MLRUN_GITHUB_REPO) \
		--tag $(MLRUN_MODELS_IMAGE_NAME) .

.PHONY: push-models
push-models: models ## Push models docker image
	docker push $(MLRUN_MODELS_IMAGE_NAME)


MLRUN_LEGACY_MODELS_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)models:$(MLRUN_DOCKER_TAG)$(MLRUN_LEGACY_DOCKER_TAG_SUFFIX)
DEFAULT_IMAGES += $(MLRUN_LEGACY_MODELS_IMAGE_NAME)

.PHONY: models-legacy
models-legacy: ## Build models legacy docker image
	docker build \
		--file dockerfiles/models/$(MLRUN_LEGACY_DOCKERFILE_DIR_NAME)/Dockerfile \
		--build-arg MLRUN_PACKAGE_TAG=$(MLRUN_PACKAGE_TAG) \
		--build-arg MLRUN_MLUTILS_GITHUB_TAG=$(MLRUN_MLUTILS_GITHUB_TAG) \
		--build-arg MLRUN_GITHUB_REPO=$(MLRUN_GITHUB_REPO) \
		--tag $(MLRUN_LEGACY_MODELS_IMAGE_NAME) .

.PHONY: push-models-legacy
push-models-legacy: models-legacy ## Push models legacy docker image
	docker push $(MLRUN_LEGACY_MODELS_IMAGE_NAME)


MLRUN_MODELS_GPU_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)models-gpu:$(MLRUN_DOCKER_TAG)
DEFAULT_IMAGES += $(MLRUN_MODELS_GPU_IMAGE_NAME)

.PHONY: modles-gpu
models-gpu: ## Build models-gpu docker image
	docker build \
		--file dockerfiles/models-gpu/Dockerfile \
		--build-arg MLRUN_PACKAGE_TAG=$(MLRUN_PACKAGE_TAG) \
		--build-arg MLRUN_MLUTILS_GITHUB_TAG=$(MLRUN_MLUTILS_GITHUB_TAG) \
		--build-arg MLRUN_GITHUB_REPO=$(MLRUN_GITHUB_REPO) \
		--tag $(MLRUN_MODELS_GPU_IMAGE_NAME) .

.PHONY: push-models-gpu
push-models-gpu: models-gpu ## Push models gpu docker image
	docker push $(MLRUN_MODELS_GPU_IMAGE_NAME)


MLRUN_LEGACY_MODELS_GPU_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)models-gpu:$(MLRUN_DOCKER_TAG)$(MLRUN_LEGACY_DOCKER_TAG_SUFFIX)
DEFAULT_IMAGES += $(MLRUN_LEGACY_MODELS_GPU_IMAGE_NAME)

.PHONY: models-gpu-legacy
models-gpu-legacy: ## Build models-gpu legacy docker image
	docker build \
		--file dockerfiles/models-gpu/$(MLRUN_LEGACY_DOCKERFILE_DIR_NAME)/Dockerfile \
		--build-arg MLRUN_PACKAGE_TAG=$(MLRUN_PACKAGE_TAG) \
		--build-arg MLRUN_MLUTILS_GITHUB_TAG=$(MLRUN_MLUTILS_GITHUB_TAG) \
		--build-arg MLRUN_GITHUB_REPO=$(MLRUN_GITHUB_REPO) \
		--tag $(MLRUN_LEGACY_MODELS_GPU_IMAGE_NAME) .

.PHONY: push-models-gpu-legacy
push-models-gpu-legacy: models-gpu-legacy ## Push models gpu legacy docker image
	docker push $(MLRUN_LEGACY_MODELS_GPU_IMAGE_NAME)


MLRUN_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/mlrun:$(MLRUN_DOCKER_TAG)
DEFAULT_IMAGES += $(MLRUN_IMAGE_NAME)

.PHONY: mlrun
mlrun: ## Build mlrun docker image
	docker build \
		--file ./Dockerfile \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		--tag $(MLRUN_IMAGE_NAME) .

.PHONY: push-mlrun
push-mlrun: mlrun ## Push mlrun docker image
	docker push $(MLRUN_IMAGE_NAME)


MLRUN_SERVING_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)serving:$(MLRUN_DOCKER_TAG)

.PHONY: serving
serving: ## Build serving docker image
	docker build \
		--file dockerfiles/serving/Dockerfile \
		--build-arg MLRUN_DOCKER_TAG=$(MLRUN_DOCKER_TAG) \
		--build-arg MLRUN_DOCKER_REPO=$(MLRUN_DOCKER_REPO) \
		--build-arg MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX=$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX) \
		--tag $(MLRUN_SERVING_IMAGE_NAME) .

.PHONY: push-serving
push-serving: serving ## Push serving docker image
	docker push $(MLRUN_SERVING_IMAGE_NAME)


MLRUN_API_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/mlrun-api:$(MLRUN_DOCKER_TAG)
DEFAULT_IMAGES += $(MLRUN_API_IMAGE_NAME)

.PHONY: api
api: ## Build mlrun-api docker image
	docker build \
		--file dockerfiles/mlrun-api/Dockerfile \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		--tag $(MLRUN_API_IMAGE_NAME) .

.PHONY: push-api
push-api: api ## Push api docker image
	docker push $(MLRUN_API_IMAGE_NAME)

MLRUN_TEST_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/test:$(MLRUN_DOCKER_TAG)

.PHONY: build-test
build-test: ## Build test docker image
	docker build \
		--file dockerfiles/test/Dockerfile \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		--tag $(MLRUN_TEST_IMAGE_NAME) .

.PHONY: push-test
push-test: build-test ## Push test docker image
	docker push $(MLRUN_TEST_IMAGE_NAME)

.PHONY: package-wheel
package-wheel: clean ## Build python package wheel
	python setup.py bdist_wheel

.PHONY: publish-package
publish-package: package-wheel ## Publish python package wheel
	python -m twine upload dist/mlrun-*.whl

.PHONY: test-publish
test-publish: package-wheel
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
		-ti \
		--rm \
		--network='host' \
		-v /tmp:/tmp \
		-v /var/run/docker.sock:/var/run/docker.sock \
		$(MLRUN_TEST_IMAGE_NAME) make test

.PHONY: test
test: clean ## Run mlrun tests
	python -m pytest -v \
		--disable-warnings \
		-rf \
		tests

.PHONY: run-api-undockerized
run-api-undockerized: ## Run mlrun api locally (un-dockerized)
	python -m mlrun db

.PHONY: docs-requirements
docs-requirements: ## Build docs requirements
	cp requirements.txt docs/requirements.txt
	echo numpydoc >> docs/requirements.txt

.PHONY: html-docs
html-docs: docs-requirements ## Build html docs
	rm -f docs/external/*.md
	cd docs && make html

.PHONY: html-docs-dockerized
html-docs-dockerized:
	docker run \
		--rm \
		-v $(PWD)/docs/_build:/mlrun/docs/_build \
		$(MLRUN_TEST_IMAGE_NAME) \
		make html-docs

.PHONY: fmt
fmt:
	@echo "Running black fmt..."
	python -m black --skip-string-normalization .

.PHONY: lint
lint: flake8 fmt-check

.PHONY: fmt-check
fmt-check:
	@echo "Running black fmt check..."
	python -m black --skip-string-normalization --check --diff -S .

.PHONY: flake8
flake8:
	@echo "Running flake8 lint..."
	python -m flake8 .

.PHONY: all help build docker-images push-docker-images print-docker-images base models models-gpu mlrun serving api \
 build-test package-wheel publish-package clean test-dockerized test run-api-undockerized docs-requirements \
 html-docs html-docs-dockerized fmt lint fmt-check flake8
