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

#
# Protobuf
#
.PHONY: build
build: compile-schemas
	@echo Done

.PHONY: compile-schemas
compile-schemas: cleanup compile-schemas-go compile-schemas-python
	@echo Done

.PHONY: compile-schemas-go
compile-schemas-go:
	protoc --go_out=. --go-grpc_out=. --go-grpc_opt=require_unimplemented_servers=false proto/*.proto

.PHONY: compile-schemas-python
compile-schemas-python:
	python -m grpc_tools.protoc --proto_path=mlrun/api/proto=proto --python_out=.. --grpc_python_out=.. proto/*.proto
	sh hack/scripts/fix_python_grpc_imports.sh

.PHONY: cleanup
cleanup:
	rm -rf schemas/build/
	rm -f ../mlrun/api/proto/*pb2*.py


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
