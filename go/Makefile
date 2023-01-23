# Copyright 2018 Iguazio
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

GO_VERSION=1.19
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
		--file cmd/logcollector/docker/Dockerfile \
		--build-arg GO_VERSION=$(GO_VERSION) \
		--tag $(MLRUN_DOCKER_IMAGE_PREFIX)/log-collector:$(MLRUN_DOCKER_TAG) \
		.

.PHONY: push-log-collector
push-log-collector:
	@echo Pushing log-collector image
	docker push $(MLRUN_DOCKER_IMAGE_PREFIX)/log-collector:$(MLRUN_DOCKER_TAG)

.PHONY: schemas-compiler
schemas-compiler: schemas-compiler
	@echo Building schemas-compiler image
	docker build \
		--file cmd/schemas_compiler/docker/Dockerfile \
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
		-v $(shell dirname $(PWD)):/app \
		$(MLRUN_DOCKER_IMAGE_PREFIX)/schemas-compiler:latest \
		make compile-schemas-local

.PHONY: compile-schemas-go
compile-schemas-go:
	@echo Compiling go schemas
	protoc --go_out=. --go-grpc_out=. --go-grpc_opt=require_unimplemented_servers=false proto/*.proto

.PHONY: compile-schemas-python
compile-schemas-python:
	@echo Compiling python schemas
	python3 -m grpc_tools.protoc --proto_path=. --python_out=../mlrun/api --grpc_python_out=../mlrun/api proto/*.proto
	@echo Fixing Python GRPC Imports
	sh hack/scripts/fix_python_grpc_imports.sh

.PHONY: cleanup
cleanup:
	rm -rf proto/build/
	rm -f ../mlrun/api/proto/*pb2*.py


#
# Test
#
.PHONY: test-unit-local
test-unit-local: compile-schemas-local
	go -race -tags=test_unit test ./... -short

.PHONY: test-integration-local
test-integration-local: compile-schemas-local
	go -race -tags=test_integration test ./...

.PHONY: test-unit-dockerized
test-unit-dockerized: schemas-compiler
	docker run \
		-v $(shell dirname $(PWD)):/app \
		$(MLRUN_DOCKER_IMAGE_PREFIX)/schemas-compiler:latest \
		make test-unit-local

.PHONY: test-integration-dockerized
test-integration-dockerized: schemas-compiler
	docker run \
		-v $(HOME)/.kube/config:/root/.kube/config \
		-v $(shell dirname $(PWD)):/app \
		$(MLRUN_DOCKER_IMAGE_PREFIX)/schemas-compiler:latest \
		make test-integration-local

#
# Misc
#
.PHONY: fmt
fmt:
	gofmt -s -w .
	golangci-lint run --fix

.PHONY: lint
lint:
	@echo Installing linters...
	@test -e $(GOPATH)/bin/impi || \
		(mkdir -p $(GOPATH)/bin && \
		curl -s https://api.github.com/repos/pavius/impi/releases/latest \
		| grep -i "browser_download_url.*impi.*$(OS_NAME)" \
		| cut -d : -f 2,3 \
		| tr -d \" \
		| wget -O $(GOPATH)/bin/impi -qi - \
		&& chmod +x $(GOPATH)/bin/impi)

	@test -e $(GOPATH)/bin/golangci-lint || \
	  	(curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(GOPATH)/bin v1.50.0)

	@echo Verifying imports...
	$(GOPATH)/bin/impi \
		--local github.com/mlrun/mlrun/ \
		--scheme stdLocalThirdParty \
		./cmd/... ./pkg/...

	@echo Linting...
	$(GOPATH)/bin/golangci-lint run -v
	@echo Done.
