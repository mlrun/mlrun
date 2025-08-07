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

GO_VERSION=$(shell grep -m1 'go ' ../go/go.mod | awk '{split($$2,a,"."); print a[1] "." a[2]}')


PROTO_DIR := proto

GO_OUT_DIR  := proto/build/go
GO_PKG_DIR  := $(GO_OUT_DIR)/$(PROTO_DIR)
PROTO_SRCS  := $(wildcard $(PROTO_DIR)/*.proto)
GO_TARGETS  := $(patsubst $(PROTO_DIR)/%.proto,\
                       $(GO_PKG_DIR)/%.pb.go,\
                       $(PROTO_SRCS))

PY_OUT_DIR   := proto/build/py
PROTO_PY_SRCS := $(wildcard $(PROTO_DIR)/*.proto)
PY_TARGETS := $(patsubst $(PROTO_DIR)/%.proto,$(PY_OUT_DIR)/%_pb2.py,$(PROTO_PY_SRCS)) \
              $(patsubst $(PROTO_DIR)/%.proto,$(PY_OUT_DIR)/%_pb2_grpc.py,$(PROTO_PY_SRCS))

PY_TARGET_DIR := ../py/schemas/proto

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

.PHONY: reset-schemas
reset-schemas: cleanup

compile-schemas-local: compile-schemas-go compile-schemas-python

.PHONY: compile-schemas-dockerized
compile-schemas-dockerized: schemas-compiler
	@echo Compiling schemas in docker container
	docker run \
		--volume $(dir $(CURDIR)):/app \
		--workdir /app/common \
		$(MLRUN_DOCKER_IMAGE_PREFIX)/schemas-compiler:latest \
		make compile-schemas-local

compile-schemas-go: $(GO_TARGETS)
	@echo "Syncing → $(GO_OUT_DIR)"
	@mkdir -p ../go/proto/build
	@cp -RL $(GO_PKG_DIR)/* ../go/proto/

compile-schemas-python: $(PY_TARGETS)
	@echo "Syncing → $(PY_TARGET_DIR)"
	@mkdir -p $(PY_TARGET_DIR)
	cp -RL $(PY_OUT_DIR)/* $(PY_TARGET_DIR)/

$(GO_PKG_DIR)/%.pb.go: $(PROTO_DIR)/%.proto
	@mkdir -p $(@D)
	@protoc \
		--go_out=$(GO_OUT_DIR) \
		--go-grpc_out=$(GO_OUT_DIR) \
		--go-grpc_opt=require_unimplemented_servers=false \
		$<

$(PY_OUT_DIR)/%_pb2.py $(PY_OUT_DIR)/%_pb2_grpc.py: $(PROTO_DIR)/%.proto
	@mkdir -p $(PY_OUT_DIR)
	python -m grpc_tools.protoc -I$(PROTO_DIR) \
	    --python_out=$(PY_OUT_DIR) \
	    --grpc_python_out=$(PY_OUT_DIR) $<

.PHONY: clean
clean:
	rm -rf $(GO_OUT_DIR) ../go/proto/build

.PHONY: cleanup
cleanup:
	@rm -rf proto/build/
	@rm -f ../py/schemas/proto/*pb2*.py
	@rm -rf ../go/proto/build/
