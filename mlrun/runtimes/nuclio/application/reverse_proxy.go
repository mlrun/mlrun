package main

import (
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"

	nuclio "github.com/nuclio/nuclio-sdk-go"
)

func Handler(context *nuclio.Context, event nuclio.Event) (interface{}, error) {
	reverseProxy := context.UserData.(map[string]interface{})["reverseProxy"].(*httputil.ReverseProxy)
	sidecarUrl := context.UserData.(map[string]interface{})["server"].(string)

	// populate reverse proxy http request
	httpRequest, err := http.NewRequest(event.GetMethod(), event.GetPath(), bytes.NewReader(event.GetBody()))
	if err != nil {
		return nil, err
	}
	recorder := httptest.NewRecorder()
	reverseProxy.ServeHTTP(recorder, httpRequest)

	// send request to sidecar
	context.Logger.InfoWith("Forwarding request to sidecar", "sidecarUrl", sidecarUrl)
	response := recorder.Result()

	headers := make(map[string]interface{})
	for key, value := range response.Header {
		headers[key] = value[0]
	}
	return nuclio.Response{
		StatusCode:  response.StatusCode,
		Body:        recorder.Body.Bytes(),
		ContentType: response.Header.Get("Content-Type"),
	}, nil
}

func InitContext(context *nuclio.Context) error {
	sidecarHost := os.Getenv("SIDECAR_HOST")
	sidecarPort := os.Getenv("SIDECAR_PORT")
	// url for request forwarding
	sidecarUrl := fmt.Sprintf("%s:%s", sidecarHost, sidecarPort)
	parsedURL, err := url.Parse(sidecarUrl)
	if err != nil {
		context.Logger.ErrorWith("Failed to parse sidecar url", "sidecarUrl", sidecarUrl)
		return err
	}
	reverseProxy := httputil.NewSingleHostReverseProxy(parsedURL)

	context.UserData = map[string]interface{}{
		"server":       sidecarUrl,
		"reverseProxy": reverseProxy,
	}
	return nil
}