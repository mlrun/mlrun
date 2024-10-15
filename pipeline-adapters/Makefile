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

PACKAGES ?= mlrun-pipelines-kfp-common mlrun-pipelines-kfp-v1-8 mlrun-pipelines-kfp-v2 mlrun-pipelines-kfp-nop

TWINE_USERNAME ?= __token__
TWINE_PASSWORD ?= "replace with your token"

.PHONY: build release clean

build:
	@for pkg in $(PACKAGES); do \
        echo "Building $$pkg"; \
        cd $$pkg && python -m build; \
        cd ..; \
    done

release:
	@set -e;\
	for pkg in $(PACKAGES); do \
        echo "Releasing $$pkg to PyPI"; \
        cd $$pkg && TWINE_USERNAME=$(TWINE_USERNAME) TWINE_PASSWORD=$(TWINE_PASSWORD) twine upload dist/*; \
        cd ..; \
    done

clean:
	@for pkg in $(PACKAGES); do \
        echo "Cleaning $$pkg"; \
        pushd $$pkg && rm -rf dist build *.egg-info src/*.egg-info && popd; \
	done
