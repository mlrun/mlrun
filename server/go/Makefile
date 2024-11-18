# Copyright 2023 Iguazio
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
MLRUN_DOCKER_TAG ?= $(shell echo "$(MLRUN_VERSION)" | sed -E 's/\+/\-/g')
MLRUN_DOCKER_IMAGE_PREFIX := $(if $(MLRUN_DOCKER_REGISTRY),$(strip $(MLRUN_DOCKER_REGISTRY))$(MLRUN_DOCKER_REPO),$(MLRUN_DOCKER_REPO))

GO_VERSION=$(shell grep -m1 'go ' go.work | cut -d' ' -f2)
GOPATH ?= $(shell go env GOPATH)
OS_NAME = $(shell uname)
KUBECONFIG := $(if $(KUBECONFIG),$(KUBECONFIG),$(HOME)/.kube/config)

CI_BIN_DIR=$(shell pwd)/.bin
GO_WORKSPACES ?= $(shell go work edit -json | jq -c -r '[.Use[].DiskPath] | map_values("./" + . + "/...")[]')
DOCKER_IMAGES = log-collector

#
# Build
#
.PHONY: build
build: docker-images
	@echo Done

.PHONY: docker-images
docker-images: compile-schemas $(DOCKER_IMAGES)
	@echo Built $(DOCKER_IMAGES)

.PHONY: log-collector
log-collector: compile-schemas
	@echo Building log-collector image
	docker build \
		--file services/log-collector/cmd/logcollector/docker/Dockerfile \
		--build-arg GO_VERSION=$(GO_VERSION) \
		--tag $(MLRUN_DOCKER_IMAGE_PREFIX)/log-collector:$(MLRUN_DOCKER_TAG) \
		.

.PHONY: push-log-collector
push-log-collector:
	@echo Pushing log-collector image
	docker push $(MLRUN_DOCKER_IMAGE_PREFIX)/log-collector:$(MLRUN_DOCKER_TAG)

.PHONY: pull-log-collector
pull-log-collector:
	docker pull $(MLRUN_DOCKER_IMAGE_PREFIX)/log-collector:$(MLRUN_DOCKER_TAG)

.PHONY: schemas-compiler
schemas-compiler:
	$(MAKE) -C ../common schemas-compiler

#
# Protobuf
#
.PHONY: compile-schemas
compile-schemas:
	$(MAKE) -C ../common compile-schemas


.PHONY: compile-schemas-local
compile-schemas-local:
	$(MAKE) -C ../common compile-schemas-local

.PHONY: compile-schemas-go
compile-schemas-go:
	$(MAKE) -C ../common compile-schemas-go

#
# Test
#
.PHONY: test-unit-local
test-unit-local: compile-schemas-local
	go test -v -race -tags=test_unit -short $(GO_WORKSPACES)

.PHONY: test-integration-local
test-integration-local: compile-schemas-local
	go test -v -p 1 -race -tags=test_integration $(GO_WORKSPACES)

.PHONY: test-unit-dockerized
test-unit-dockerized: schemas-compiler
	docker run \
		--volume $(dir $(CURDIR)):/app \
		--workdir /app/go \
		$(MLRUN_DOCKER_IMAGE_PREFIX)/schemas-compiler:latest \
		make test-unit-local

.PHONY: test-integration-dockerized
test-integration-dockerized: schemas-compiler
	MLRUN_TEST_KUBECONFIG=$(if $(MLRUN_TEST_KUBECONFIG),$(MLRUN_TEST_KUBECONFIG),$(KUBECONFIG)) \
	docker run \
		--network host \
		--volume $(dir $(CURDIR)):/app \
		--volume $(MLRUN_TEST_KUBECONFIG)/:/kubeconfig \
		--env MINIKUBE_HOME=$(MINIKUBE_HOME) \
		--env KUBECONFIG=/kubeconfig \
		--workdir /app/go \
		$(MLRUN_DOCKER_IMAGE_PREFIX)/schemas-compiler:latest \
		make test-integration-local

#
# Misc
#
.PHONY: fmt
fmt: ensure-golangci
	@echo Formatting...
	gofmt -s -w .
	$(CI_BIN_DIR)/golangci-lint run --fix -- $(GO_WORKSPACES)
	@echo Done.

.PHONY: lint
lint: ensure-golangci compile-schemas-go ensure-test-files-annotated
	@echo Linting...
	$(CI_BIN_DIR)/golangci-lint run --verbose -- $(GO_WORKSPACES)
	@echo Done.

.PHONY: ensure-test-files-annotated
ensure-test-files-annotated:
	$(eval test_files_missing_build_annotations=$(strip $(shell find . -type f -name '*_test.go' -exec bash -c "grep -m 1 -L '//go:build ' {} | grep go" \;)))
	@if [ -n "$(test_files_missing_build_annotations)" ]; then \
		echo "Found go test files without build annotations: "; \
		echo $(test_files_missing_build_annotations); \
		echo "!!! Go test files must be annotated with '//go:build test_<x>' !!!"; \
		exit 1; \
	fi
	@echo "All go test files have //go:build test_X annotation"
	@exit $(.SHELLSTATUS)

.PHONY: ensure-golangci
ensure-golangci:
	@test -e $(CI_BIN_DIR)/golangci-lint || \
		(curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(CI_BIN_DIR) v1.55.2)
