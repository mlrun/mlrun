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

ARG PYTHON_VERSION=3.9
ARG GO_VERSION=1.21

FROM gcr.io/iguazio/golang:${GO_VERSION}-alpine AS golang

FROM gcr.io/iguazio/python:${PYTHON_VERSION}-alpine

ARG PROTOC_GEN_GO_VERSION=v1.33
ARG PROTOC_GEN_GO_GRPC_VERSION=v1.5.1
ARG GRPCIO_TOOLS_VERSION="~=1.59.0"

WORKDIR /app/log-collector

RUN apk --no-cache upgrade && apk add --no-cache protoc build-base linux-headers

COPY --from=golang /usr/local/go/ /usr/local/go/

# add copied golang binary to path, add go bin to path (where we install go binaries)
ENV PATH="/usr/local/go/bin:/root/go/bin:${PATH}"

RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@${PROTOC_GEN_GO_VERSION} && \
  go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@${PROTOC_GEN_GO_GRPC_VERSION}

RUN pip install --upgrade setuptools
RUN pip install grpcio-tools${GRPCIO_TOOLS_VERSION}
