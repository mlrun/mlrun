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
MLRUN_PACKAGE_TAG ?= development
MLRUN_GITHUB_REPO ?= mlrun
MLRUN_PYTHON_VERSION ?= 3.7
MLRUN_LEGACY_ML_PYTHON_VERSION ?= 3.6
MLRUN_MLUTILS_GITHUB_TAG ?= development


MLRUN_DOCKER_IMAGE_PREFIX := $(if $(MLRUN_DOCKER_REGISTRY),$(strip $(MLRUN_DOCKER_REGISTRY))$(MLRUN_DOCKER_REPO),$(MLRUN_DOCKER_REPO))
MLRUN_LEGACY_DOCKER_TAG_SUFFIX := -py$(subst .,,$(MLRUN_LEGACY_ML_PYTHON_VERSION))
MLRUN_LEGACY_DOCKERFILE_DIR_NAME := py$(subst .,,$(MLRUN_LEGACY_ML_PYTHON_VERSION))


help: ## Display available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


all:
	$(error please pick a target)


build: docker-images package-wheel ## Build all artifacts
	@echo Done.


DOCKER_IMAGES_RULES = \
	api \
	base \
	models \
	models-gpu \
	mlrun

docker-images: $(DOCKER_IMAGES_RULES) ## Build all docker images
	@echo Done.


push-docker-images: docker-images ## Push all docker images
	@for image in $(IMAGES_TO_PUSH); do \
		echo "Pushing $$image" ; \
		docker push $$image ; \
	done
	@echo Done.


print-docker-images: ## Print all docker images
	@for image in $(IMAGES_TO_PUSH); do \
		echo $$image ; \
	done


MLRUN_BASE_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)base:$(MLRUN_DOCKER_TAG)
MLRUN_LEGACY_BASE_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)base:$(MLRUN_DOCKER_TAG)$(MLRUN_LEGACY_DOCKER_TAG_SUFFIX)

base: ## Build base and legacy-base docker images
	docker build \
		--file dockerfiles/base/Dockerfile \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		--build-arg MLRUN_PACKAGE_TAG=$(MLRUN_PACKAGE_TAG) \
		--build-arg MLRUN_MLUTILS_GITHUB_TAG=${MLRUN_MLUTILS_GITHUB_TAG} \
		--build-arg MLRUN_GITHUB_REPO=$(MLRUN_GITHUB_REPO) \
		--tag $(MLRUN_BASE_IMAGE_NAME) .

	docker build \
		--file dockerfiles/base/$(MLRUN_LEGACY_DOCKERFILE_DIR_NAME)/Dockerfile \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_LEGACY_ML_PYTHON_VERSION) \
		--build-arg MLRUN_PACKAGE_TAG=$(MLRUN_PACKAGE_TAG) \
		--build-arg MLRUN_MLUTILS_GITHUB_TAG=$(MLRUN_MLUTILS_GITHUB_TAG) \
		--build-arg MLRUN_GITHUB_REPO=$(MLRUN_GITHUB_REPO) \
		--tag $(MLRUN_LEGACY_BASE_IMAGE_NAME) .

IMAGES_TO_PUSH += $(MLRUN_BASE_IMAGE_NAME)
IMAGES_TO_PUSH += $(MLRUN_LEGACY_BASE_IMAGE_NAME)


MLRUN_MODELS_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)models:$(MLRUN_DOCKER_TAG)
MLRUN_LEGACY_MODELS_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)models:$(MLRUN_DOCKER_TAG)$(MLRUN_LEGACY_DOCKER_TAG_SUFFIX)

models: ## Build models and legacy-models docker images
	docker build \
		--file dockerfiles/models/Dockerfile \
		--build-arg MLRUN_PACKAGE_TAG=$(MLRUN_PACKAGE_TAG) \
		--build-arg MLRUN_MLUTILS_GITHUB_TAG=$(MLRUN_MLUTILS_GITHUB_TAG) \
		--build-arg MLRUN_GITHUB_REPO=$(MLRUN_GITHUB_REPO) \
		--tag $(MLRUN_MODELS_IMAGE_NAME) .

	docker build \
		--file dockerfiles/models/$(MLRUN_LEGACY_DOCKERFILE_DIR_NAME)/Dockerfile \
		--build-arg MLRUN_PACKAGE_TAG=$(MLRUN_PACKAGE_TAG) \
		--build-arg MLRUN_MLUTILS_GITHUB_TAG=$(MLRUN_MLUTILS_GITHUB_TAG) \
		--build-arg MLRUN_GITHUB_REPO=$(MLRUN_GITHUB_REPO) \
		--tag $(MLRUN_LEGACY_MODELS_IMAGE_NAME) .

IMAGES_TO_PUSH += $(MLRUN_MODELS_IMAGE_NAME)
IMAGES_TO_PUSH += $(MLRUN_LEGACY_MODELS_IMAGE_NAME)


MLRUN_MODELS_GPU_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)models-gpu:$(MLRUN_DOCKER_TAG)
MLRUN_LEGACY_MODELS_GPU_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)models-gpu:$(MLRUN_DOCKER_TAG)$(MLRUN_LEGACY_DOCKER_TAG_SUFFIX)

models-gpu: ## Build models-gpu and legacy-models-gpu docker images
	docker build \
		--file dockerfiles/models-gpu/Dockerfile \
		--build-arg MLRUN_PACKAGE_TAG=$(MLRUN_PACKAGE_TAG) \
		--build-arg MLRUN_MLUTILS_GITHUB_TAG=$(MLRUN_MLUTILS_GITHUB_TAG) \
		--build-arg MLRUN_GITHUB_REPO=$(MLRUN_GITHUB_REPO) \
		--tag $(MLRUN_MODELS_GPU_IMAGE_NAME) .

	docker build \
		--file dockerfiles/models-gpu/$(MLRUN_LEGACY_DOCKERFILE_DIR_NAME)/Dockerfile \
		--build-arg MLRUN_PACKAGE_TAG=$(MLRUN_PACKAGE_TAG) \
		--build-arg MLRUN_MLUTILS_GITHUB_TAG=$(MLRUN_MLUTILS_GITHUB_TAG) \
		--build-arg MLRUN_GITHUB_REPO=$(MLRUN_GITHUB_REPO) \
		--tag $(MLRUN_LEGACY_MODELS_GPU_IMAGE_NAME) .

IMAGES_TO_PUSH += $(MLRUN_MODELS_GPU_IMAGE_NAME)
IMAGES_TO_PUSH += $(MLRUN_LEGACY_MODELS_GPU_IMAGE_NAME)


MLRUN_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/mlrun:$(MLRUN_DOCKER_TAG)

mlrun: ## Build mlrun docker image
	docker build \
		--file ./Dockerfile \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		--tag $(MLRUN_IMAGE_NAME) .

IMAGES_TO_PUSH += $(MLRUN_IMAGE_NAME)


MLRUN_SERVING_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX)serving:$(MLRUN_DOCKER_TAG)

serving: ## Build serving docker image
	docker build \
		--file dockerfiles/serving/Dockerfile \
		--build-arg MLRUN_DOCKER_TAG=$(MLRUN_DOCKER_TAG) \
		--build-arg MLRUN_DOCKER_REPO=$(MLRUN_DOCKER_REPO) \
		--build-arg MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX=$(MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX) \
		--tag $(MLRUN_SERVING_IMAGE_NAME) .


MLRUN_API_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/mlrun-api:$(MLRUN_DOCKER_TAG)

api: ## Build mlrun-api docker image
	docker build \
		--file dockerfiles/mlrun-api/Dockerfile \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		--tag $(MLRUN_API_IMAGE_NAME) .

IMAGES_TO_PUSH += $(MLRUN_API_IMAGE_NAME)


MLRUN_TEST_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/test:$(MLRUN_DOCKER_TAG)

build-test: ## Build test docker image
	docker build \
		--file dockerfiles/test/Dockerfile \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		--tag $(MLRUN_TEST_IMAGE_NAME) .


package-wheel: clean ## Build python package wheel
	python setup.py bdist_wheel


publish-package: package-wheel ## Publish python package wheel
	python -m twine upload dist/mlrun-*.whl


clean: ## Clean python package build artifacts
	rm -rf build
	rm -rf dist
	rm -rf mlrun.egg-info
	find . -name '*.pyc' -exec rm {} \;


test-dockerized: build-test ## Run mlrun tests in docker container
	docker run \
		-ti \
		--rm \
		--network='host' \
		-v /tmp:/tmp \
		-v /var/run/docker.sock:/var/run/docker.sock \
		$(MLRUN_TEST_IMAGE_NAME) make test


test: clean ## Run mlrun tests
	python -m pytest -v \
		--disable-warnings \
		-rf \
		tests


run-api-undockerized: ## Run mlrun api locally (un-dockerized)
	python -m mlrun db


circleci: test-dockerized
	docker run \
		--rm \
		-v $(PWD)/docs/_build:/mlrun/docs/_build \
		$(MLRUN_TEST_IMAGE_NAME) \
		make html-docs


docs-requirements: ## Build docs requirements
	cp requirements.txt docs/requirements.txt
	echo numpydoc >> docs/requirements.txt


html-docs: docs-requirements ## Build html docs
	rm -f docs/external/*.md
	cd docs && make html


.PHONY: all help build docker-images push-docker-images print-docker-images base models models-gpu mlrun serving api \
 build-test package-wheel publish-package clean test-dockerized test run-api-undockerized circleci docs-requirements \
 html-docs
