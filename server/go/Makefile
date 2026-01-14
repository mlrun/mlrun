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

GO_VERSION=$(shell grep -m1 'go ' go.mod | awk '{split($$2,a,"."); print a[1] "." a[2]}')
GOPATH ?= $(shell go env GOPATH)
OS_NAME = $(shell uname)
KUBECONFIG := $(if $(KUBECONFIG),$(KUBECONFIG),$(HOME)/.kube/config)

CI_BIN_DIR=$(shell pwd)/.bin
DOCKER_IMAGES = log-collector
LOG_COLLECTOR_IMAGE_NAME := $(MLRUN_DOCKER_IMAGE_PREFIX)/log-collector:$(MLRUN_DOCKER_TAG)
DEFAULT_IMAGES += $(LOG_COLLECTOR_IMAGE_NAME)

GOLANGCI_LINT_VERSION := 2.4.0
GOLANGCI_LINT_BIN := $(CI_BIN_DIR)/golangci-lint
GOLANGCI_LINT_INSTALL_COMMAND := GOBIN=$(CI_BIN_DIR) go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v$(GOLANGCI_LINT_VERSION)

GOTESTSUM_VERSION = v1.13.0
GOTESTSUM_BIN := $(CI_BIN_DIR)/gotestsum-$(GOTESTSUM_VERSION)

#
# Build
#
.PHONY: build
build: docker-images
	@echo Done

.PHONY: docker-images
docker-images: compile-schemas $(DOCKER_IMAGES)
	@echo Built $(DOCKER_IMAGES)

.PHONY: print-docker-images
print-docker-images:
	@for image in $(DEFAULT_IMAGES); do \
		echo $$image ; \
	done

.PHONY: log-collector
log-collector: compile-schemas
	@echo Building log-collector image
	docker build \
		--file services/logcollector/cmd/docker/Dockerfile \
		--build-arg GO_VERSION=$(GO_VERSION) \
		--tag $(LOG_COLLECTOR_IMAGE_NAME) \
		.

.PHONY: push-log-collector
push-log-collector:
	@echo Pushing log-collector image
	docker push $(LOG_COLLECTOR_IMAGE_NAME)

.PHONY: pull-log-collector
pull-log-collector:
	docker pull $(LOG_COLLECTOR_IMAGE_NAME)

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

$(GOTESTSUM_BIN):
	@echo "$(YELLOW)Installing gotestsum $(GOTESTSUM_VERSION)...$(NC)"
	GOBIN=$(CI_BIN_DIR) go install gotest.tools/gotestsum@$(GOTESTSUM_VERSION) && \
    mv $(CI_BIN_DIR)/gotestsum $(GOTESTSUM_BIN)

.PHONY: test-unit-local
test-unit-local: compile-schemas-local $(GOTESTSUM_BIN)
	$(GOTESTSUM_BIN) --format testname -- -v -race -tags=test_unit -short ./...

.PHONY: test-integration-local
test-integration-local: compile-schemas-local $(GOTESTSUM_BIN)
	$(GOTESTSUM_BIN) --format testname -- -v -p 1 -race -tags=test_integration ./...

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
fmt: ensure-golangci compile-schemas-go
	@echo Formatting...
	gofmt -s -w .
	$(GOLANGCI_LINT_BIN) run --fix ./...
	@echo Done.

.PHONY: lint
lint: ensure-golangci compile-schemas-go ensure-test-files-annotated
	@echo Linting...
	$(GOLANGCI_LINT_BIN) run --verbose ./...
	@echo Done.

.PHONY: security
security: compile-schemas-go
	@echo Running security checks...
	go run golang.org/x/vuln/cmd/govulncheck@latest ./...


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
	@if ! command -v $(GOLANGCI_LINT_BIN) >/dev/null 2>&1; then \
		echo "golangci-lint not found. Installing..."; \
		$(GOLANGCI_LINT_INSTALL_COMMAND); \
	else \
		installed_version=$$($(GOLANGCI_LINT_BIN) version | awk '/version/ {print $$4}' | sed 's/^v//'); \
		if [ "$$installed_version" != "$(GOLANGCI_LINT_VERSION)" ]; then \
			echo "golangci-lint version mismatch ($$installed_version != $(GOLANGCI_LINT_VERSION)). Reinstalling..."; \
			$(GOLANGCI_LINT_INSTALL_COMMAND); \
		fi \
	fi
