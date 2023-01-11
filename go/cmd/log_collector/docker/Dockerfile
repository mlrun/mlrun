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
ARG GRPC_HEALTH_PROBE_VERSION=v0.4.14

FROM golang:${GO_VERSION} as build-binary

RUN mkdir /app
WORKDIR /app

# copy `go.mod` for definitions and `go.sum` to invalidate the next layer
# in case of a change in the dependencies
COPY go.mod go.sum /app/

RUN go mod download

ADD . /app/
RUN CGO_ENABLED=0 GOARCH=amd64 GOOS=linux go build -a -installsuffix cgo -o main cmd/log_collector/main.go

RUN wget -qO/app/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/v0.4.14/grpc_health_probe-linux-amd64 && \
    chmod +x /app/grpc_health_probe

FROM scratch
COPY --from=build-binary /app/main /main
COPY --from=build-binary /app/grpc_health_probe /grpc_health_probe

EXPOSE 8080

CMD ["/main"]
