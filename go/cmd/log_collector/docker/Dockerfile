ARG GO_VERSION=1.19.4

FROM golang:${GO_VERSION} as build-binary

RUN mkdir /app
ADD . /app/
WORKDIR /app
RUN CGO_ENABLED=0 GOARCH=amd64 GOOS=linux go build -a -installsuffix cgo -o main cmd/log_collector/main.go

FROM scratch
COPY --from=build-binary /app/main /main

EXPOSE 8080

CMD ["/main"]