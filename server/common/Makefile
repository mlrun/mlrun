# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
MLRUN_VERSION ?= unstable
MLRUN_DOCKER_REGISTRY ?=
MLRUN_DOCKER_REPO ?= mlrun
MLRUN_DOCKER_IMAGE_PREFIX := $(if $(MLRUN_DOCKER_REGISTRY),$(strip $(MLRUN_DOCKER_REGISTRY))$(MLRUN_DOCKER_REPO),$(MLRUN_DOCKER_REPO))

GO_VERSION=$(shell grep -m1 'go ' ../go/go.work | cut -d' ' -f2)

#
# Build
#

.PHONY: schemas-compiler
schemas-compiler:
	@echo Building schemas-compiler image
	docker build \
		--platform linux/amd64 \
		--file builder/schemas_compiler/docker/Dockerfile \
		--build-arg GO_VERSION=$(GO_VERSION) \
		--tag $(MLRUN_DOCKER_IMAGE_PREFIX)/schemas-compiler:latest \
		.
#
# Protobuf
#
.PHONY: compile-schemas
compile-schemas: compile-schemas-dockerized
	@echo Done

.PHONY: compile-schemas-local
compile-schemas-local: cleanup compile-schemas-go compile-schemas-python
	@echo Done

.PHONY: compile-schemas-dockerized
compile-schemas-dockerized: schemas-compiler
	@echo Compiling schemas in docker container
	docker run \
		--volume $(dir $(CURDIR)):/app \
		--workdir /app/common \
		$(MLRUN_DOCKER_IMAGE_PREFIX)/schemas-compiler:latest \
		make compile-schemas-local

.PHONY: compile-schemas-go
compile-schemas-go: cleanup
	@echo Compiling go schemas
	mkdir -p proto/tmp
	protoc --go_out=proto/tmp --go-grpc_out=proto/tmp --go-grpc_opt=require_unimplemented_servers=false proto/*.proto
	mkdir -p proto/build/go
	mv proto/tmp/proto/build/* proto/build/go
	rm -rf proto/tmp
	mkdir -p ../go/mlrun-go/pkg/proto/build
	cp -RL proto/build/go/* ../go/mlrun-go/pkg/proto/build

.PHONY: compile-schemas-python
compile-schemas-python:
	@echo Compiling python schemas
	python3 -m grpc_tools.protoc --proto_path=. --python_out=proto/build/ --grpc_python_out=proto/build/ proto/*.proto
	@echo Fixing Python GRPC Dir and Imports
	sh hack/scripts/fix_python_grpc_imports.sh
	cp -RL proto/build/py/* ../py/services/api/proto

.PHONY: cleanup
cleanup:
	rm -rf proto/build/
	rm -f ../py/services/api/proto/*pb2*.py
	rm -rf ../go/mlrun-go/pkg/proto/build/
