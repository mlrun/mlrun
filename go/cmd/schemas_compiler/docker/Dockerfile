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
ARG GO_VERSION=1.19
FROM golang:${GO_VERSION}
ARG PROTOC_GEN_GO_VERSION=v1.28
ARG PROTOC_GEN_GO_GRPC_VERSION=v1.2
ARG GRPCIO_TOOLS_VERSION="~=1.41.0"
WORKDIR /app/go
RUN apt-get update && apt install -y \
    protobuf-compiler \
    python3 \
    python3-setuptools  \
    python3-pip
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@${PROTOC_GEN_GO_VERSION} && \
  go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@${PROTOC_GEN_GO_GRPC_VERSION}
RUN pip install grpcio-tools${GRPCIO_TOOLS_VERSION}