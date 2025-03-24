// Copyright 2024 Iguazio
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package main

import (
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"

	nuclio "github.com/nuclio/nuclio-sdk-go"
)

func Handler(context *nuclio.Context, event nuclio.Event) (interface{}, error) {
	reverseProxy := context.UserData.(map[string]interface{})["reverseProxy"].(*httputil.ReverseProxy)
	sidecarUrl := context.UserData.(map[string]interface{})["server"].(string)

	// populate reverse proxy http request
	httpRequest, err := http.NewRequest(event.GetMethod(), event.GetPath(), bytes.NewReader(event.GetBody()))
	if err != nil {
		context.Logger.ErrorWith("Failed to create a reverse proxy request")
		return nil, err
	}
	for k, v := range event.GetHeaders() {
		httpRequest.Header[k] = []string{v.(string)}
	}

	// populate query params
	query := httpRequest.URL.Query()
	for k, v := range event.GetFields() {
		query.Set(k, v.(string))
	}
	httpRequest.URL.RawQuery = query.Encode()

	recorder := httptest.NewRecorder()
	reverseProxy.ServeHTTP(recorder, httpRequest)

	// send request to sidecar
	context.Logger.DebugWith("Forwarding request to sidecar",
		"sidecarUrl", sidecarUrl,
		"method", event.GetMethod())
	response := recorder.Result()

	headers := make(map[string]interface{})
	for key, value := range response.Header {
		headers[key] = value[0]
	}

	// let the processor calculate the content length
	delete(headers, "Content-Length")
	return nuclio.Response{
		StatusCode:  response.StatusCode,
		Body:        recorder.Body.Bytes(),
		ContentType: response.Header.Get("Content-Type"),
		Headers:     headers,
	}, nil
}

func InitContext(context *nuclio.Context) error {
	sidecarHost := os.Getenv("SIDECAR_HOST")
	sidecarPort := os.Getenv("SIDECAR_PORT")
	if sidecarHost == "" {
		sidecarHost = "http://localhost"
	} else if !strings.Contains(sidecarHost, "://") {
		sidecarHost = fmt.Sprintf("http://%s", sidecarHost)
	}

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
