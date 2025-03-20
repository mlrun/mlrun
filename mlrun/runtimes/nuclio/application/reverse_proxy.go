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
	"regexp"
	"strings"

	nuclio "github.com/nuclio/nuclio-sdk-go"
)

var (
	ansiEscape    = regexp.MustCompile(`\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])`)
	controlChars  = regexp.MustCompile(`[\x00-\x1F\x7F-\x9F]`)
	malformedAnsi = regexp.MustCompile(`\[\d{1,2};\d{1,2}m`)
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
	sanitizedQuery := sanitizeQuery(httpRequest.URL.Query())
	context.Logger.DebugWith("Forwarding request to sidecar",
		"sidecarUrl", sidecarUrl,
		"method", event.GetMethod(),
		"query", sanitizedQuery)
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

// sanitizeQuery sanitizes the query params to prevent log injection
func sanitizeQuery(query url.Values) url.Values {
	sanitizedQuery := url.Values{}
	for k, v := range query {
		sanitizedValues := make([]string, len(v))
		for i, value := range v {
			sanitizedValue := sanitizeQueryParam(value)
			sanitizedValues[i] = sanitizedValue
		}
		sanitizedQuery[k] = sanitizedValues
	}
	return sanitizedQuery
}

// sanitizeQueryParam sanitizes a single query parameter value by removing control characters and ANSI escape sequences
func sanitizeQueryParam(value string) string {
	// Replace newlines with spaces
	value = strings.ReplaceAll(value, "\n", " ")
	value = strings.ReplaceAll(value, "\r", " ")

	// Remove ANSI escape sequences (full support for various patterns)
	value = ansiEscape.ReplaceAllString(value, "")

	// Remove all ASCII control characters (0-31, 127-159)
	value = controlChars.ReplaceAllString(value, "")

	// Remove malformed ANSI-like sequences (e.g., "value5[0;31m")
	value = malformedAnsi.ReplaceAllString(value, "")

	return value
}
