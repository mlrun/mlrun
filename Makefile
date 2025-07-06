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
# mainly used for mlrun and mlrun-gpu. mlrun API version >= 1.3.0 should always have python 3.9
MLRUN_PYTHON_VERSION ?= 3.11
PYTHON_VERSION ?= $(shell python --version)
MLRUN_SKIP_COMPILE_SCHEMAS ?=
INCLUDE_PYTHON_VERSION_SUFFIX ?=
MLRUN_PIP_VERSION ?= 25.0.0
MLRUN_UV_VERSION ?= 0.7.14
MLRUN_UV_IMAGE ?= ghcr.io/astral-sh/uv:$(MLRUN_UV_VERSION)
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
MLRUN_SYSTEM_TEST_MARKERS ?=
MLRUN_SYSTEM_TESTS_GITHUB_RUN_URL ?=
MLRUN_GPU_CUDA_VERSION ?= 12.8.1-cudnn-devel-ubuntu22.04
RUN_COVERAGE ?= false
COVERAGE_FILE ?=
COVERAGE_MOUNT_PATH ?=
ifeq ("$(RUN_COVERAGE)","true")
    COVERAGE_ADDITION = -m coverage run --data-file=$$COVERAGE_FILE
else
    COVERAGE_ADDITION =
endif

SETUP_COVERAGE = if [ "$(RUN_COVERAGE)" = "true" ]; then \
	case "$$COVERAGE_FILE" in *.coverage) \
		rm -rf $$COVERAGE_FILE && \
		mkdir -p $$(dirname $$COVERAGE_FILE) ;\
		;; \
	  *) \
		echo "Error: COVERAGE_FILE must end with .coverage" >&2; \
		exit 1; \
		;; \
	esac \
fi

PRINT_COVERAGE_REPORT = if [ "$(RUN_COVERAGE)" = "true" ]; then \
    	echo "coverage report $$COVERAGE_FILE :"; \
		COVERAGE_FILE=$$COVERAGE_FILE coverage report; \
	fi

# Verify the mount point to avoid deleting essential paths
SETUP_COVERAGE_MOUNTING = if [ "$(RUN_COVERAGE)" = "true" ]; then \
		case "$$COVERAGE_MOUNT_PATH" in /tmp/coverage_reports/*) \
			rm -rf $$COVERAGE_MOUNT_PATH && \
			mkdir -p $$COVERAGE_MOUNT_PATH; \
			;; \
	  	*) \
			echo "Error: COVERAGE_MOUNT_PATH is invalid, must be under /tmp/coverage_reports/*" >&2 ; \
			exit 1; \
			;; \
		esac \
	fi
# THIS BLOCK IS FOR COMPUTED VARIABLES
MLRUN_DOCKER_IMAGE_PREFIX := $(if $(MLRUN_DOCKER_REGISTRY),$(strip $(MLRUN_DOCKER_REGISTRY))$(MLRUN_DOCKER_REPO),$(MLRUN_DOCKER_REPO))
MLRUN_CACHE_DOCKER_IMAGE_PREFIX := $(if $(MLRUN_DOCKER_CACHE_FROM_REGISTRY),$(strip $(MLRUN_DOCKER_CACHE_FROM_REGISTRY))$(MLRUN_DOCKER_REPO),$(MLRUN_DOCKER_REPO))
# if MLRUN_NO_CACHE passed we don't want to use cache, this is mainly used for cleaner if statements
MLRUN_USE_CACHE := $(if $(MLRUN_NO_CACHE),,true)
MLRUN_DOCKER_NO_CACHE_FLAG := $(if $(MLRUN_NO_CACHE),--no-cache,)
MLRUN_PIP_NO_CACHE_FLAG := $(if $(MLRUN_NO_CACHE),--no-cache-dir,)
# expected to be in the form of '-py<major><minor>' e.g. '-py39'
MLRUN_ANACONDA_PYTHON_DISTRIBUTION := $(shell echo "$(MLRUN_PYTHON_VERSION)" | awk -F. '{print "-py"$$1$$2}')
MLRUN_PYTHON_VERSION_SUFFIX := $(if $(INCLUDE_PYTHON_VERSION_SUFFIX),$(MLRUN_ANACONDA_PYTHON_DISTRIBUTION),)

# expected to be in the form of 'py<major><minor>' e.g. 'py39'
MLRUN_LINT_PYTHON_VERSION := $(shell echo "$(MLRUN_PYTHON_VERSION)" | awk -F. '{print "py"$$1$$2}')

MLRUN_PIPELINES_KFP_VERSION := $(if $(filter 3.9,$(MLRUN_PYTHON_VERSION)),1-8,2)

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

MLRUN_PYTHON_PACKAGE_INSTALLER ?= pip
ifeq ($(MLRUN_PYTHON_PACKAGE_INSTALLER),pip)
	MLRUN_PYTHON_VENV_PIP_INSTALL ?= python -m pip install
else ifeq ($(MLRUN_PYTHON_PACKAGE_INSTALLER),uv)
	MLRUN_PYTHON_VENV_PIP_INSTALL ?= uv pip install --python-version $(MLRUN_PYTHON_VERSION)
else
	$(error MLRUN_PYTHON_PACKAGE_INSTALLER must be either "pip" or "uv")
endif

# Change to `--upgrade-package <package-name>` to upgrade only a specific package
MLRUN_UV_UPGRADE_FLAG ?= --upgrade

# absolute path to this Makefile
THIS_MAKEFILE := $(abspath $(lastword $(MAKEFILE_LIST)))
# its directory
ROOT_DIR       := $(dir   $(THIS_MAKEFILE))

.PHONY: help
help: ## Display available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


.PHONY: all
all:
	$(error please pick a target)

.PHONY: install-requirements
install-requirements: ## Install all requirements needed for development
	# relevant for pip package installer only
	@if [ "$(MLRUN_PYTHON_PACKAGE_INSTALLER)" = "pip" ]; then \
		$(MLRUN_PYTHON_VENV_PIP_INSTALL) --upgrade $(MLRUN_PIP_NO_CACHE_FLAG) pip~=$(MLRUN_PIP_VERSION); \
	fi

	$(MLRUN_PYTHON_VENV_PIP_INSTALL) \
		$(MLRUN_PIP_NO_CACHE_FLAG) \
		-r requirements.txt \
		-r extras-requirements.txt \
		-r dev-requirements.txt \
		-r dockerfiles/mlrun-api/requirements.txt

.PHONY: install-dev-requirements
install-dev-requirements: ## Install dev-requirements relevant for pytest and coverage.
	# relevant for pip package installer only
	@if [ "$(MLRUN_PYTHON_PACKAGE_INSTALLER)" = "pip" ]; then \
		$(MLRUN_PYTHON_VENV_PIP_INSTALL) --upgrade $(MLRUN_PIP_NO_CACHE_FLAG) pip~=$(MLRUN_PIP_VERSION); \
	fi

	$(MLRUN_PYTHON_VENV_PIP_INSTALL) \
		$(MLRUN_PIP_NO_CACHE_FLAG) \
		-r dev-requirements.txt

.PHONY: install-docs-requirements
install-docs-requirements: ## Install all requirements needed for compiling mlrun docs
	$(MLRUN_PYTHON_VENV_PIP_INSTALL) --upgrade $(MLRUN_PIP_NO_CACHE_FLAG) pip~=$(MLRUN_PIP_VERSION)
	$(MLRUN_PYTHON_VENV_PIP_INSTALL) $(MLRUN_PIP_NO_CACHE_FLAG) -r docs/requirements.txt

.PHONY: install-conda-requirements
install-conda-requirements: ## Install all requirements needed for development with specific conda packages for arm64
ifeq ($(findstring 3.11.,$(PYTHON_VERSION)),3.11.)
	conda install --yes --file conda-arm64-requirements-python311.txt
else ifeq ($(findstring 3.9.,$(PYTHON_VERSION)),3.9.)
	conda install --yes --file conda-arm64-requirements-python39.txt
else
	@echo "Unsupported Python version: $(PYTHON_VERSION)" >&2
	@exit 1
endif
	make install-requirements

.PHONY: install-complete-requirements
install-complete-requirements: ## Install all requirements needed for development and testing
	$(MLRUN_PYTHON_VENV_PIP_INSTALL) --upgrade $(MLRUN_PIP_NO_CACHE_FLAG) pip~=$(MLRUN_PIP_VERSION)
	$(eval MLRUN_PIP_INSTALL_FLAG := $(if $(and $(MLRUN_PYTHON_PACKAGE_INSTALLER),$(filter -m pip,$(MLRUN_PYTHON_PACKAGE_INSTALLER))),--ignore-requires-python,))
	$(MLRUN_PYTHON_VENV_PIP_INSTALL) .[complete,dev-postgres] $(MLRUN_PIP_INSTALL_FLAG)

.PHONY: install-complete-kfp-requirements
install-complete-kfp-requirements: ## Install all requirements needed for development and testing + KFP 1.8
	$(MLRUN_PYTHON_VENV_PIP_INSTALL) --upgrade $(MLRUN_PIP_NO_CACHE_FLAG) pip~=$(MLRUN_PIP_VERSION)
	$(eval MLRUN_PIP_INSTALL_FLAG := $(if $(and $(MLRUN_PYTHON_PACKAGE_INSTALLER),$(filter -m pip,$(MLRUN_PYTHON_PACKAGE_INSTALLER))),--ignore-requires-python,))
	$(MLRUN_PYTHON_VENV_PIP_INSTALL) .[complete,kfp18,dev-postgres] $(MLRUN_PIP_INSTALL_FLAG)

.PHONY: install-all-requirements
install-all-requirements: ## Install all requirements needed for development and testing
	$(MLRUN_PYTHON_VENV_PIP_INSTALL) --upgrade $(MLRUN_PIP_NO_CACHE_FLAG) pip~=$(MLRUN_PIP_VERSION)
	$(eval MLRUN_PIP_INSTALL_FLAG := $(if $(and $(MLRUN_PYTHON_PACKAGE_INSTALLER),$(filter -m pip,$(MLRUN_PYTHON_PACKAGE_INSTALLER))),--ignore-requires-python,))
	$(MLRUN_PYTHON_VENV_PIP_INSTALL) .[all] $(MLRUN_PIP_INSTALL_FLAG)

.PHONY: create-migration-mysql
create-migration-mysql: ## Create a DB migration (MLRUN_MIGRATION_MESSAGE must be set)
	./automation/scripts/create_migration_mysql.sh

.PHONY: create-migration
create-migration: create-migration-mysql
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

.PHONY: generate-dockerignore
generate-dockerignore: ## Copies the root .dockerignore and removes the tests pattern from it
	$(eval TARGET := dockerfiles/${DEST}/Dockerfile.dockerignore)
	@if [ -f "$(TARGET)" ]; then \
		temp_file=$$(mktemp) && \
		sed '/\*\*\/tests/d' .dockerignore > $$temp_file && \
		if cmp -s $$temp_file "$(TARGET)"; then \
			echo "File $(TARGET) already exists and content is identical"; \
			rm $$temp_file; \
			exit 0; \
		else \
			echo "File $(TARGET) exists but content differs, updating..."; \
			mv $$temp_file "$(TARGET)"; \
		fi; \
	else \
		sed '/\*\*\/tests/d' .dockerignore > "$(TARGET)"; \
	fi


.PHONY: build
build: docker-images package-wheel ## Build all artifacts
	@echo Done.

DEFAULT_DOCKER_IMAGES_RULES = \
	api \
	mlrun \
	mlrun-gpu \
	mlrun-kfp \
	jupyter \
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
		--build-arg MLRUN_UV_IMAGE=$(MLRUN_UV_IMAGE) \
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


MLRUN_KFP_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/mlrun-kfp
MLRUN_KFP_CACHE_IMAGE_NAME := $(MLRUN_CACHE_DOCKER_IMAGE_PREFIX)/mlrun-kfp
MLRUN_KFP_IMAGE_NAME_TAGGED := $(MLRUN_KFP_IMAGE_NAME):$(MLRUN_DOCKER_TAG)$(MLRUN_PYTHON_VERSION_SUFFIX)
MLRUN_KFP_CACHE_IMAGE_NAME_TAGGED := $(MLRUN_KFP_CACHE_IMAGE_NAME):$(MLRUN_DOCKER_CACHE_FROM_TAG)$(MLRUN_PYTHON_VERSION_SUFFIX)
MLRUN_KFP_IMAGE_DOCKER_CACHE_FROM_FLAG := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_USE_CACHE)),--cache-from $(strip $(MLRUN_KFP_CACHE_IMAGE_NAME_TAGGED)),)
MLRUN_KFP_CACHE_IMAGE_PULL_COMMAND := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_USE_CACHE)), docker pull $(MLRUN_CACHE_IMAGE_NAME_TAGGED) || true,)
MLRUN_KFP_CACHE_IMAGE_PUSH_COMMAND := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_PUSH_DOCKER_CACHE_IMAGE)),docker tag $(MLRUN_KFP_IMAGE_NAME_TAGGED) $(MLRUN_KFP_CACHE_IMAGE_NAME_TAGGED) && docker push $(MLRUN_KFP_CACHE_IMAGE_NAME_TAGGED),)

DEFAULT_IMAGES += $(MLRUN_KFP_IMAGE_NAME_TAGGED)

.PHONY: mlrun-kfp
mlrun-kfp: update-version-file ## Build mlrun docker image with KFP
	$(MLRUN_KFP_CACHE_IMAGE_PULL_COMMAND)
	docker build \
		--file dockerfiles/mlrun-kfp/Dockerfile \
		--build-arg MLRUN_DOCKER_REGISTRY=$(MLRUN_DOCKER_REGISTRY) \
		--build-arg MLRUN_VERSION=$(MLRUN_VERSION) \
		--build-arg MLRUN_PIP_VERSION=$(MLRUN_PIP_VERSION) \
		$(MLRUN_KFP_IMAGE_DOCKER_CACHE_FROM_FLAG) \
		$(MLRUN_DOCKER_NO_CACHE_FLAG) \
		--tag $(MLRUN_KFP_IMAGE_NAME):$(MLRUN_DOCKER_TAG)$(MLRUN_PYTHON_VERSION_SUFFIX) .

.PHONY: push-mlrun-kfp
push-mlrun-kfp: mlrun-kfp ## Push mlrun docker image
	docker push $(MLRUN_KFP_IMAGE_NAME_TAGGED)
	$(MLRUN_KFP_CACHE_IMAGE_PUSH_COMMAND)

.PHONY: pull-mlrun-kfp
pull-mlrun-kfp: ## Pull mlrun docker image
	docker pull $(MLRUN_KFP_CACHE_IMAGE_PULL_COMMAND)

MLRUN_GPU_PREBAKED_IMAGE_NAME_TAGGED := quay.io/mlrun/prebaked-cuda:$(MLRUN_GPU_CUDA_VERSION)
MLRUN_GPU_PREBAKED_PY39_IMAGE_NAME_TAGGED := quay.io/mlrun/prebaked-cuda:11.8.0-cudnn8-devel-ubuntu22.04
MLRUN_GPU_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/mlrun-gpu
MLRUN_GPU_CACHE_IMAGE_NAME := $(MLRUN_CACHE_DOCKER_IMAGE_PREFIX)/mlrun-gpu
MLRUN_GPU_IMAGE_NAME_TAGGED := $(MLRUN_GPU_IMAGE_NAME):$(MLRUN_DOCKER_TAG)$(MLRUN_PYTHON_VERSION_SUFFIX)
# Choose the GPU base image based on the minor Python version
MLRUN_GPU_BASE_IMAGE ?= $(shell \
  PY_MINOR=$$(echo "$(MLRUN_PYTHON_VERSION)" | cut -d. -f2); \
  if [ "$$PY_MINOR" = "9" ]; then \
    echo "$(MLRUN_GPU_PREBAKED_PY39_IMAGE_NAME_TAGGED)"; \
  else \
    echo "$(MLRUN_GPU_PREBAKED_IMAGE_NAME_TAGGED)"; \
  fi \
)
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
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		--build-arg MLRUN_GPU_BASE_IMAGE=$(MLRUN_GPU_BASE_IMAGE) \
		--build-arg MLRUN_UV_IMAGE=$(MLRUN_UV_IMAGE) \
		--build-arg MLRUN_PIP_VERSION=$(MLRUN_PIP_VERSION) \
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
		--build-arg MLRUN_PIP_VERSION=$(MLRUN_PIP_VERSION) \
		--tag $(MLRUN_GPU_PREBAKED_IMAGE_NAME_TAGGED) \
		.

.PHONY: push-prebake-mlrun-gpu
push-prebake-mlrun-gpu: ## Push prebake mlrun GPU based docker image
	docker push $(MLRUN_GPU_PREBAKED_IMAGE_NAME_TAGGED)

MLRUN_JUPYTER_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/jupyter
MLRUN_JUPYTER_CACHE_IMAGE_NAME := $(MLRUN_CACHE_DOCKER_IMAGE_PREFIX)/jupyter
MLRUN_JUPYTER_IMAGE_NAME_TAGGED := $(MLRUN_JUPYTER_IMAGE_NAME):$(MLRUN_DOCKER_TAG)$(MLRUN_PYTHON_VERSION_SUFFIX)
MLRUN_JUPYTER_CACHE_IMAGE_NAME_TAGGED := $(MLRUN_JUPYTER_CACHE_IMAGE_NAME):$(MLRUN_DOCKER_CACHE_FROM_TAG)$(MLRUN_PYTHON_VERSION_SUFFIX)
MLRUN_JUPYTER_IMAGE_DOCKER_CACHE_FROM_FLAG := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_USE_CACHE)),--cache-from $(strip $(MLRUN_JUPYTER_CACHE_IMAGE_NAME_TAGGED)),)
MLRUN_JUPYTER_CACHE_IMAGE_PUSH_COMMAND := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_PUSH_DOCKER_CACHE_IMAGE)),docker tag $(MLRUN_JUPYTER_IMAGE_NAME_TAGGED) $(MLRUN_JUPYTER_CACHE_IMAGE_NAME_TAGGED) && docker push $(MLRUN_JUPYTER_CACHE_IMAGE_NAME_TAGGED),)
MLRUN_JUPYTER_CACHE_IMAGE_PULL_COMMAND := $(if $(and $(MLRUN_DOCKER_CACHE_FROM_TAG),$(MLRUN_USE_CACHE)),docker pull $(MLRUN_JUPYTER_CACHE_IMAGE_NAME_TAGGED) || true,)
DEFAULT_IMAGES += $(MLRUN_JUPYTER_IMAGE_NAME_TAGGED)

.PHONY: jupyter
jupyter: update-version-file ## Build mlrun jupyter docker image
	$(MLRUN_JUPYTER_CACHE_IMAGE_PULL_COMMAND)
	docker build \
		--file dockerfiles/jupyter/Dockerfile \
		--build-arg MLRUN_PIP_VERSION=$(MLRUN_PIP_VERSION) \
		--build-arg MLRUN_CACHE_DATE=$(MLRUN_CACHE_DATE) \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		--build-arg MLRUN_UV_IMAGE=$(MLRUN_UV_IMAGE) \
		$(MLRUN_JUPYTER_IMAGE_DOCKER_CACHE_FROM_FLAG) \
		$(MLRUN_DOCKER_NO_CACHE_FLAG) \
		--tag $(MLRUN_JUPYTER_IMAGE_NAME_TAGGED) \
		.

.PHONY: push-jupyter
push-jupyter: jupyter ## Push mlrun jupyter docker image
	docker push $(MLRUN_JUPYTER_IMAGE_NAME_TAGGED)
	$(MLRUN_JUPYTER_CACHE_IMAGE_PUSH_COMMAND)

.PHONY: pull-jupyter
pull-jupyter: ## Pull mlrun jupyter docker image
	docker pull $(MLRUN_JUPYTER_IMAGE_NAME_TAGGED)

.PHONY: log-collector
log-collector: update-version-file
	@MLRUN_VERSION=$(MLRUN_VERSION) \
		MLRUN_DOCKER_REGISTRY=$(MLRUN_DOCKER_REGISTRY) \
		MLRUN_DOCKER_REPO=$(MLRUN_DOCKER_REPO) \
		MLRUN_DOCKER_TAG=$(MLRUN_DOCKER_TAG) \
		MLRUN_DOCKER_IMAGE_PREFIX=$(MLRUN_DOCKER_IMAGE_PREFIX) \
		make --no-print-directory -C $(shell pwd)/server/go log-collector

.PHONY: push-log-collector
push-log-collector: log-collector
	@MLRUN_VERSION=$(MLRUN_VERSION) \
		MLRUN_DOCKER_REGISTRY=$(MLRUN_DOCKER_REGISTRY) \
		MLRUN_DOCKER_REPO=$(MLRUN_DOCKER_REPO) \
		MLRUN_DOCKER_TAG=$(MLRUN_DOCKER_TAG) \
		MLRUN_DOCKER_IMAGE_PREFIX=$(MLRUN_DOCKER_IMAGE_PREFIX) \
		make --no-print-directory -C $(shell pwd)/server/go push-log-collector

.PHONY: pull-log-collector
pull-log-collector:
	@MLRUN_VERSION=$(MLRUN_VERSION) \
		MLRUN_DOCKER_REGISTRY=$(MLRUN_DOCKER_REGISTRY) \
		MLRUN_DOCKER_REPO=$(MLRUN_DOCKER_REPO) \
		MLRUN_DOCKER_TAG=$(MLRUN_DOCKER_TAG) \
		MLRUN_DOCKER_IMAGE_PREFIX=$(MLRUN_DOCKER_IMAGE_PREFIX) \
		make --no-print-directory -C $(shell pwd)/server/go pull-log-collector


.PHONY: compile-schemas
compile-schemas: ## Compile schemas over docker
ifdef MLRUN_SKIP_COMPILE_SCHEMAS
	@echo "Skipping compile schemas"
else
	$(MAKE) -C server/go compile-schemas
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
		--build-arg MLRUN_UV_IMAGE=$(MLRUN_UV_IMAGE) \
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
	$(MAKE) generate-dockerignore DEST=test
	$(MLRUN_TEST_CACHE_IMAGE_PULL_COMMAND)
	docker build \
		--file dockerfiles/test/Dockerfile \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		--build-arg MLRUN_PIP_VERSION=$(MLRUN_PIP_VERSION) \
		--build-arg MLRUN_PIPELINES_KFP_VERSION=$(MLRUN_PIPELINES_KFP_VERSION) \
		--build-arg MLRUN_UV_VERSION=$(MLRUN_UV_VERSION) \
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
	$(MAKE) generate-dockerignore DEST=test-system
	docker build \
		--file dockerfiles/test-system/Dockerfile \
		--build-arg MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		--build-arg MLRUN_PIP_VERSION=$(MLRUN_PIP_VERSION) \
		--build-arg MLRUN_UV_VERSION=$(MLRUN_UV_VERSION) \
		$(MLRUN_DOCKER_NO_CACHE_FLAG) \
		--tag $(MLRUN_SYSTEM_TEST_IMAGE_NAME) .

.PHONY: package-wheel
package-wheel: clean update-version-file ## Build python package wheel
	uv build

.PHONY: publish-package
publish-package: package-wheel ## Publish python package wheel
	uv publish

.PHONY: test-publish
test-publish: package-wheel ## Test python package publishing
	uv publish --publish-url https://test.pypi.org/legacy/

.PHONY: clean
clean: ## Clean python package build artifacts
	rm -rf build dist mlrun.egg-info
	find . -type f -name '*.pyc' ! -path './venv/*' -delete

.PHONY: test-dockerized
test-dockerized: build-test ## Run mlrun tests in docker container
	COVERAGE_MOUNT_PATH="/tmp/coverage_reports/unit_tests$(COVERAGE_DIR_SUFFIX)" ;\
	$(SETUP_COVERAGE_MOUNTING) && \
	docker run \
		-t \
		--rm \
		--network='host' \
		-e MLRUN_PYTHON_VERSION=$(MLRUN_PYTHON_VERSION) \
		-v /tmp:/tmp \
		-v $$COVERAGE_MOUNT_PATH:/mlrun/tests/coverage_reports \
		-v /var/run/docker.sock:/var/run/docker.sock \
		$(MLRUN_TEST_IMAGE_NAME_TAGGED) make test  UNIT_TESTS_IGNORE_PATH="$(UNIT_TESTS_IGNORE_PATH)" \
		UNIT_TESTS_PATH="$(UNIT_TESTS_PATH)" \
		RUN_COVERAGE=$(RUN_COVERAGE) \
		COVERAGE_FILE="$(COVERAGE_FILE)"


.PHONY: test
test: clean ## Run mlrun tests
	# TODO: Remove ignored tests for Python 3.11 compatibility with KFP 2
	set -e ; \
	COMMON_IGNORE_TEST_FLAGS=$$(echo "\
	--ignore=tests/integration \
	--ignore=server/py/services/api/tests/integration \
	--ignore=tests/system \
	--ignore=tests/rundb/test_httpdb.py \
	--ignore=server/py/services/api/migrations \
	") && \
	PER_PYTHON_VERSION_IGNORE_TEST_FLAGS=$(if $(filter $(MLRUN_PYTHON_VERSION),3.11),$$(echo "\
		--ignore=server/py/services/api/tests/unit/api/test_pipelines.py \
		--ignore=tests/projects/test_kfp.py \
		--ignore=server/py/services/api/tests/unit/crud/test_pipelines.py \
		--ignore=tests/serving/test_remote.py \
		--ignore=tests/projects/test_remote_pipeline.py \
		--ignore=pipeline-adapters/mlrun-pipelines-kfp-v1-8/tests \
		"),) && \
	if [ "$(UNIT_TESTS_IGNORE_PATH)" != "" ]; then \
  		IGNORE_ADDITION="--ignore=$(UNIT_TESTS_IGNORE_PATH)"; \
	else \
		IGNORE_ADDITION=""; \
	fi && \
	COVERAGE_FILE=$(COVERAGE_FILE) && \
	COVERAGE_FILE=$${COVERAGE_FILE:-"tests/coverage_reports/unit_tests.coverage"} && \
	$(SETUP_COVERAGE) && \
	python \
		-X faulthandler \
		$(COVERAGE_ADDITION) \
		-m pytest -v \
		--capture=no \
		--disable-warnings \
		--durations=100 \
		$$COMMON_IGNORE_TEST_FLAGS \
		$$PER_PYTHON_VERSION_IGNORE_TEST_FLAGS \
		$$IGNORE_ADDITION \
		--forked \
		-rf \
		$$UNIT_TESTS_PATH && \
	$(PRINT_COVERAGE_REPORT) ;



.PHONY: test-integration-dockerized
test-integration-dockerized: build-test ## Run mlrun integration tests in docker container
	COVERAGE_MOUNT_PATH="/tmp/coverage_reports/integration_tests" ;\
	$(SETUP_COVERAGE_MOUNTING)  && \
	docker run \
		-t \
		--rm \
		--network='host' \
		-v /tmp:/tmp \
		-v /var/run/docker.sock:/var/run/docker.sock \
		-v $$COVERAGE_MOUNT_PATH:/mlrun/tests/coverage_reports \
		-e RUN_COVERAGE=$(RUN_COVERAGE) \
		--add-host=host.docker.internal:host-gateway \
		$(MLRUN_TEST_IMAGE_NAME_TAGGED) make test-integration

.PHONY: test-integration
test-integration: clean ## Run mlrun integration tests
	set -e; \
	COVERAGE_FILE=$(COVERAGE_FILE) && \
	COVERAGE_FILE=$${COVERAGE_FILE:-"tests/coverage_reports/integration_tests.coverage"} && \
	$(SETUP_COVERAGE) && \
	python $(COVERAGE_ADDITION) \
		-m pytest -v \
		--capture=no \
		--disable-warnings \
		--durations=100 \
		-rf \
		tests/integration \
		server/py/services/api/tests/integration \
		tests/rundb/test_httpdb.py && \
	$(PRINT_COVERAGE_REPORT);

.PHONY: test-migrations-dockerized
test-migrations-dockerized: build-test ## Run mlrun db migrations tests in docker container
	COVERAGE_MOUNT_PATH="/tmp/coverage_reports/migration_tests" ;\
	$(SETUP_COVERAGE_MOUNTING) && \
	docker run \
		-t \
		--rm \
		--network='host' \
		-v $(shell pwd):/mlrun \
		-v /tmp:/tmp \
		-v /var/run/docker.sock:/var/run/docker.sock \
		-e RUN_COVERAGE=$(RUN_COVERAGE) \
		-v $$COVERAGE_MOUNT_PATH:/mlrun/tests/coverage_reports \
		$(MLRUN_TEST_IMAGE_NAME_TAGGED) make RUN_COVERAGE=true test-migrations

.PHONY: test-migrations
test-migrations: clean ## Run mlrun db migrations tests
	COVERAGE_FILE=$(COVERAGE_FILE) && \
	COVERAGE_FILE=$${COVERAGE_FILE:-"tests/coverage_reports/migration_tests.coverage"} && \
	export COVERAGE_FILE && \
	$(SETUP_COVERAGE) && \
	bash -c 'set -euo pipefail; \
	  python -u $(COVERAGE_ADDITION) -m pytest -vvv \
	    --capture=no --disable-warnings --durations=100 \
	    -rf "$(ROOT_DIR)/server/py/services/api/migrations/tests" \
	    2>&1 | tee migration_tests.log' ; \
	exit_code=$$? ; \
	$(PRINT_COVERAGE_REPORT) ; \
	exit $$exit_code

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
	MLRUN_SYSTEM_TESTS_CLEAN_RESOURCES=$(MLRUN_SYSTEM_TESTS_CLEAN_RESOURCES) \
	MLRUN_SYSTEM_TESTS_GITHUB_RUN_URL=$(MLRUN_SYSTEM_TESTS_GITHUB_RUN_URL) \
	python  \
		-m pytest -v \
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
		-m $(if $(MLRUN_SYSTEM_TEST_MARKERS),"$(MLRUN_SYSTEM_TEST_MARKERS)","not enterprise") \
		$(MLRUN_SYSTEM_TESTS_COMMAND_SUFFIX)

.PHONY: test-package compile-schemas
test-package: ## Run mlrun package tests
	python ./automation/package_test/test.py run

.PHONY: test-go
test-go-unit: ## Run mlrun go unit tests
	$(MAKE) -C server/go test-unit-local

.PHONY: test-go-dockerized
test-go-unit-dockerized: ## Run mlrun go unit tests in docker container
	$(MAKE) -C server/go test-unit-dockerized

.PHONY: test-go
test-go-integration: ## Run mlrun go unit tests
	$(MAKE) -C server/go test-integration-local

.PHONY: test-go-dockerized
test-go-integration-dockerized: ## Run mlrun go integration tests in docker container
	$(MAKE) -C server/go test-integration-dockerized

.PHONY: run-api-undockerized
run-api-undockerized: ## Run mlrun api locally (un-dockerized)
	python -m mlrun db

.PHONY: run-api
run-api: api ## Run mlrun api (dockerized)
	# clean up any previous api container. Don't remove it after run to be able to debug failures
	docker rm mlrun-api --force || true
	docker run \
		--name mlrun-api \
		--detach \
		--publish 8080 \
		--add-host host.docker.internal:host-gateway \
		--env MLRUN_HTTPDB__DSN=$(MLRUN_HTTPDB__DSN) \
		--env MLRUN_LOG_LEVEL=$(MLRUN_LOG_LEVEL) \
		--env MLRUN_LOG_FORMATTER=$(MLRUN_LOG_FORMATTER) \
		--env MLRUN_SECRET_STORES__TEST_MODE_MOCK_SECRETS=$(MLRUN_SECRET_STORES__TEST_MODE_MOCK_SECRETS) \
		--env MLRUN_HTTPDB__REAL_PATH=$(MLRUN_HTTPDB__REAL_PATH) \
		$(MLRUN_API_IMAGE_NAME_TAGGED)

.PHONY: run-test-db
run-test-db:
	# clean up any previous test db container. Don't remove it after run to be able to debug failures
	docker rm test-db --force || true
	docker run \
		--name=test-db \
		--volume $(shell pwd):/mlrun \
		--publish 3306:3306 \
		--env MYSQL_ROOT_PASSWORD="" \
		--env MYSQL_ALLOW_EMPTY_PASSWORD="true" \
		--env MYSQL_ROOT_HOST=% \
		--env MYSQL_DATABASE="mlrun" \
		--detach \
		gcr.io/iguazio/mlrun-mysql:8.0 \
		--character-set-server=utf8 \
		--collation-server=utf8_bin

.PHONY: clean-html-docs
clean-html-docs: ## Clean html docs
	rm -f docs/external/*.md
	make -C docs clean

.PHONY: html-docs
html-docs: clean-html-docs ## Build html docs
	make -C docs html

.PHONY: html-docs-dockerized
html-docs-dockerized: build-test ## Build html docs dockerized
	docker run \
		--rm \
		-v $(shell pwd)/docs/_build:/mlrun/docs/_build \
		$(MLRUN_TEST_IMAGE_NAME_TAGGED) \
		bash -c 'make install-docs-requirements && make html-docs'

.PHONY: fmt
fmt: ## Format the code using Ruff and blacken-docs
	@echo "Running ruff checks and fixes..."
	python -m ruff check --fix-only
	python -m ruff format
	@echo "Formatting the code blocks with blacken-docs..."
	git ls-files -z -- '*.md' | xargs -0 blacken-docs -t="$(MLRUN_LINT_PYTHON_VERSION)"

.PHONY: lint-docs
lint-docs: ## Format the code blocks in markdown files
	@echo "Checking the code blocks with blacken-docs"
	git ls-files -z -- '*.md' | xargs -0 blacken-docs -t="$(MLRUN_LINT_PYTHON_VERSION)" --check
	@if [ "$(SKIP_VALE_CHECK)" != "true" ]; then \
	    make vale-docs; \
	fi

.PHONY: lint-imports
lint-imports: ## Validates import dependencies
	@echo "Running import linter"
	lint-imports

.PHONY: lint
lint: lint-check lint-imports ## Run lint on the code

.PHONY: lint-check
lint-check: ## Check the code (using ruff)
	@echo "Running ruff checks..."
	python -m ruff check --exit-non-zero-on-fix
	python -m ruff check --preview --select=CPY001 --exit-non-zero-on-fix
	python -m ruff format --check

.PHONY: lint-go
lint-go:
	$(MAKE) -C server/go lint

.PHONY: security-go
security-go:
	$(MAKE) -C server/go security

.PHONY: fmt-go
fmt-go:
	$(MAKE) -C server/go fmt

.PHONY: vale-docs
vale-docs: ## Run vale check for docs and sorts ignore.txt file
	vale docs
	@sort .github/styles/MLRun/ignore.txt -o .github/styles/MLRun/ignore.txt

.PHONY: linkcheck
linkcheck:
	make -C docs/ linkcheck

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
	# Run tests for the base code
	export MLRUN_HTTPDB__DSN='sqlite:////mlrun/db/mlrun.db?check_same_thread=false' && \
	export MLRUN_OPENAPI_JSON_NAME=mlrun_bc_base_oai.json && \
	cd $(MLRUN_BC_TESTS_BASE_CODE_PATH) && \
	pip install ./pipeline-adapters/mlrun-pipelines-kfp-common && \
	pip install ./pipeline-adapters/mlrun-pipelines-kfp-v1-8 && \
	python -m pytest -v --capture=no --disable-warnings --durations=100 server/py/services/api/tests/unit/api/test_docs.py::test_save_openapi_json && \
	cd ..

	# Run tests for the head code (feature branch)
	export MLRUN_OPENAPI_JSON_NAME=mlrun_bc_head_oai.json && \
	pip install ./pipeline-adapters/mlrun-pipelines-kfp-common && \
	pip install ./pipeline-adapters/mlrun-pipelines-kfp-v1-8 && \
	python -m pytest -v --capture=no --disable-warnings --durations=100 server/py/services/api/tests/unit/api/test_docs.py::test_save_openapi_json

	# Run OpenAPI diff to check compatibility
	docker run --rm -t -v $(MLRUN_BC_TESTS_OPENAPI_OUTPUT_PATH):/specs:ro openapitools/openapi-diff:latest /specs/mlrun_bc_base_oai.json /specs/mlrun_bc_head_oai.json --fail-on-incompatible


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
	python ./automation/release_notes/generate.py \
		run \
		--release $(MLRUN_VERSION) \
		--previous-release $(MLRUN_OLD_VERSION) \
		--release-branch $(MLRUN_RELEASE_BRANCH) \
		--raise-on-failed-parsing $(MLRUN_RAISE_ON_ERROR) \
		--tmp-file-path $(MLRUN_RELEASE_NOTES_OUTPUT_FILE) \
		--skip-clone $(MLRUN_SKIP_CLONE)


.PHONY: pull-cache
pull-cache: ## Pull images to be used as cache for build
ifdef MLRUN_DOCKER_CACHE_FROM_TAG
	targets="$(subst push-,,$(MAKECMDGOALS))" ; \
	for image_name in $$targets; do \
		tag=$(MLRUN_DOCKER_CACHE_FROM_TAG)$(MLRUN_PYTHON_VERSION_SUFFIX) ; \
		docker pull $(MLRUN_CACHE_DOCKER_IMAGE_PREFIX)/$$image_name:$$tag || true ; \
	done;
endif


.PHONY: upgrade-mlrun-api-deps-lock
upgrade-mlrun-api-deps-lock: ## Upgrade mlrun-api locked requirements file
	uv pip compile \
		requirements.txt \
		extras-requirements.txt \
		dockerfiles/mlrun-api/requirements.txt \
		$(MLRUN_UV_UPGRADE_FLAG) \
		--python-version $(MLRUN_PYTHON_VERSION) \
		--output-file dockerfiles/mlrun-api/locked-requirements.txt

.PHONY: upgrade-mlrun-mlrun-deps-lock
upgrade-mlrun-mlrun-deps-lock: ## Upgrade mlrun-mlrun locked requirements file
	uv pip compile \
		requirements.txt \
		extras-requirements.txt \
		dockerfiles/mlrun/requirements.txt \
		$(MLRUN_UV_UPGRADE_FLAG) \
		--output-file dockerfiles/mlrun/locked-requirements.txt

.PHONY: upgrade-mlrun-gpu-deps-lock
upgrade-mlrun-gpu-deps-lock: ## Upgrade mlrun-gpu locked requirements file
	uv pip compile \
		requirements.txt \
		extras-requirements.txt \
		dockerfiles/mlrun/requirements.txt \
		$(MLRUN_UV_UPGRADE_FLAG) \
		--output-file dockerfiles/gpu/locked-requirements.txt

.PHONY: upgrade-mlrun-jupyter-deps-lock
upgrade-mlrun-jupyter-deps-lock: ## Upgrade mlrun-jupyter locked requirements file
	uv pip compile \
		requirements.txt \
		extras-requirements.txt \
		dockerfiles/jupyter/requirements.txt \
		$(MLRUN_UV_UPGRADE_FLAG) \
		--python-version $(MLRUN_PYTHON_VERSION) \
		--output-file dockerfiles/jupyter/locked-requirements.txt

.PHONY: upgrade-mlrun-test-deps-lock
upgrade-mlrun-test-deps-lock: ## Upgrade mlrun test locked requirements file
	uv pip compile \
		requirements.txt \
		extras-requirements.txt \
		dockerfiles/mlrun-api/requirements.txt \
		dockerfiles/test/requirements.txt \
		dev-requirements.txt \
		$(MLRUN_UV_UPGRADE_FLAG) \
		--output-file dockerfiles/test/locked-requirements.txt

.PHONY: upgrade-mlrun-system-test-deps-lock
upgrade-mlrun-system-test-deps-lock: ## Upgrade mlrun system test locked requirements file
	uv pip compile \
		requirements.txt \
		extras-requirements.txt \
		dockerfiles/mlrun-kfp/requirements.txt \
		dockerfiles/mlrun-api/requirements.txt \
		dev-requirements.txt \
		$(MLRUN_UV_UPGRADE_FLAG) \
		--output-file dockerfiles/test-system/locked-requirements.txt

upgrade-mlrun-kfp-deps-lock: ## Upgrade mlrun-kfp locked requirements file
	uv pip compile \
		requirements.txt \
		dockerfiles/mlrun-kfp/requirements.txt \
		--python-version 3.9 \
		$(MLRUN_UV_UPGRADE_FLAG) \
		--output-file dockerfiles/mlrun-kfp/locked-requirements.txt

.PHONY: upgrade-mlrun-deps-lock
upgrade-mlrun-deps-lock: ## Upgrade mlrun-* locked requirements file
	@$(MAKE) -j \
		upgrade-mlrun-mlrun-deps-lock \
		upgrade-mlrun-api-deps-lock \
		upgrade-mlrun-jupyter-deps-lock \
		upgrade-mlrun-gpu-deps-lock \
		upgrade-mlrun-kfp-deps-lock \
		upgrade-mlrun-test-deps-lock \
		upgrade-mlrun-system-test-deps-lock


.PHONY: coverage-combine
coverage-combine: ## Combine all coverage reports, ignoring errors like missing or corrupted source files
	rm -f tests/coverage_reports/combined.coverage; \
	UNIT_TEST_COVERAGE_PATHS=$${UNIT_TEST_COVERAGE_PATHS:-"tests/coverage_reports/unit_tests.coverage"}; \
	COVERAGE_FILE=tests/coverage_reports/combined.coverage coverage combine --keep \
	$$UNIT_TEST_COVERAGE_PATHS \
	tests/coverage_reports/integration_tests.coverage \
	tests/coverage_reports/migration_tests.coverage; \
	python -m coverage xml --ignore-errors --data-file=tests/coverage_reports/combined.coverage -o tests/coverage_reports/combined.xml; \
	echo "Full coverage report:"; \
	COVERAGE_FILE=tests/coverage_reports/combined.coverage coverage report -i
