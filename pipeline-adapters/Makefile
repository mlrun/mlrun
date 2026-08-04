# Copyright 2024 Iguazio
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

PACKAGES ?= mlrun-pipelines-kfp-common mlrun-pipelines-kfp-v1-8 mlrun-pipelines-kfp-v2

TWINE_USERNAME ?= __token__
TWINE_PASSWORD ?= "replace with your token"

BUMP ?= patch

.DEFAULT_GOAL := help

.PHONY: help build release clean bump-version

help: ## Display available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

bump-version: ## Bump the patch (default) or minor (BUMP=minor) version across all packages
	@python bump_version.py $(BUMP)

build: ## Build all packages
	@for pkg in $(PACKAGES); do \
        echo "Building $$pkg"; \
        cd $$pkg && python -m build; \
        cd ..; \
    done

release: ## Release all packages to PyPI
	@set -e;\
	for pkg in $(PACKAGES); do \
        echo "Releasing $$pkg to PyPI"; \
        cd $$pkg && TWINE_USERNAME=$(TWINE_USERNAME) TWINE_PASSWORD=$(TWINE_PASSWORD) twine upload dist/*; \
        cd ..; \
    done

clean: ## Remove build artifacts from all packages
	@for pkg in $(PACKAGES); do \
	  echo "Cleaning $$pkg"; \
	  (cd $$pkg && rm -rf dist build *.egg-info src/*.egg-info); \
	done
