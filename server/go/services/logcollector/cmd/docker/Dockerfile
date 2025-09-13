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

ARG GO_VERSION=1.25

FROM gcr.io/iguazio/golang:${GO_VERSION} AS build-binary

RUN mkdir /app
WORKDIR /app

# Build stages requires defining GO_VERSION again
ARG GO_VERSION

# copy `go.mod` for definitions and `go.sum` to invalidate the next layer
# in case of a change in the dependencies
COPY go.mod go.sum ./

RUN go mod download

COPY framework ./framework
COPY services ./services
COPY proto ./proto

# build
RUN GOOS=linux \
    GOARCH=amd64 \
    CGO_ENABLED=0 \
    go build -a -installsuffix cgo -ldflags="-s -w" -o logcollector ./services/logcollector/cmd/main.go


FROM gcr.io/distroless/static-debian12

COPY --from=build-binary /app/logcollector /main

EXPOSE 8080

VOLUME /var/mlrun

CMD [ "/main" ]
