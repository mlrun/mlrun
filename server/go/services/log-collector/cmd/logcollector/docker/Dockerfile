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

ARG GO_VERSION=1.21

FROM gcr.io/iguazio/golang:${GO_VERSION} AS build-binary

RUN mkdir /app
WORKDIR /app

# Build stages requires defining GO_VERSION again
ARG GO_VERSION=1.21

# copy `go.mod` for definitions and `go.sum` to invalidate the next layer
# in case of a change in the dependencies
COPY mlrun-go/go.mod mlrun-go/go.sum ./mlrun-go/
COPY services/log-collector/go.mod services/log-collector/go.sum ./services/log-collector/

RUN echo "go ${GO_VERSION}\n use(\n\t mlrun-go \n\t services/log-collector \n)" > go.work && \
    cat go.work && \
    go work sync

RUN go mod download

COPY mlrun-go/ ./mlrun-go/
COPY services/log-collector/ ./services/log-collector/

WORKDIR /app/services/log-collector

# build
RUN GOOS=linux \
    GOARCH=amd64 \
    CGO_ENABLED=0 \
    go build -a -installsuffix cgo -ldflags="-s -w" -o main cmd/logcollector/main.go

FROM alpine:latest AS install-health-probe

ARG GRPC_HEALTH_PROBE_VERSION=v0.4.28

RUN mkdir /app
WORKDIR /app

RUN wget -qO/app/grpc_health_probe \
    https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 && \
    chmod +x /app/grpc_health_probe

FROM gcr.io/iguazio/alpine:3.20

# Upgrade OS packages
RUN apk -U upgrade

COPY --from=build-binary /app/services/log-collector/main /main
COPY --from=install-health-probe /app/grpc_health_probe /grpc_health_probe

EXPOSE 8080

VOLUME /var/mlrun

CMD [ "/main" ]
