// Copyright 2018 Iguazio
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package common

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/nuclio/errors"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
)

func GetEnvOrDefaultString(key string, defaultValue string) string {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	} else if value == "nil" || value == "none" {
		return ""
	}
	return value
}

func GetEnvOrDefaultBool(key string, defaultValue bool) bool {
	return strings.ToLower(GetEnvOrDefaultString(key, strconv.FormatBool(defaultValue))) == "true"
}

func GetEnvOrDefaultInt(key string, defaultValue int) int {
	valueInt, err := strconv.Atoi(GetEnvOrDefaultString(key, strconv.Itoa(defaultValue)))
	if err != nil {
		return defaultValue
	}
	return valueInt
}

func GetKubernetesClientConfig(kubeconfigPath string) (*rest.Config, error) {
	if kubeconfigPath != "" {
		return clientcmd.BuildConfigFromFlags("", kubeconfigPath)
	}

	return rest.InClusterConfig()
}

func EnsureDirExists(dirPath string, mode os.FileMode) error {
	if _, err := os.Stat(dirPath); os.IsNotExist(err) {
		if err := os.MkdirAll(dirPath, mode); err != nil {
			return err
		}
	}

	return nil
}

func EnsureFileExists(filePath string) error {
	if _, err := os.Stat(filePath); os.IsNotExist(err) {

		// get file directory
		dirPath := filepath.Dir(filePath)
		if err := os.MkdirAll(dirPath, 0755); err != nil {
			return errors.Wrapf(err, "Failed to create directory - %s", dirPath)
		}
		if _, err := os.Create(filePath); err != nil {
			return errors.Wrapf(err, "Failed to create file - %s", filePath)
		}
	}

	return nil
}

// WriteToFile writes the given bytes to the given file path
func WriteToFile(filePath string,
	content []byte,
	append bool) error {

	// this flag enables us to create the file if it doesn't exist, and open the file read/write permissions
	openFlags := os.O_CREATE | os.O_RDWR
	if append {
		openFlags = openFlags | os.O_APPEND
	} else {

		// if we're not appending, we want to truncate the file
		openFlags = openFlags | os.O_TRUNC
	}

	if err := EnsureFileExists(filePath); err != nil {
		return errors.Wrap(err, "Failed to ensure file exists")
	}

	// open file
	file, err := os.OpenFile(filePath, openFlags, 0644)
	if err != nil {
		return errors.Wrapf(err, "Failed to open file - %s", filePath)
	}

	defer file.Close() // nolint: errcheck

	if _, err := file.Write(content); err != nil {
		return errors.Wrapf(err, "Failed to write log contents to file - %s", filePath)
	}

	return nil
}

func SyncMapLength(m *sync.Map) int {
	var i int
	m.Range(func(k, v interface{}) bool {
		i++
		return true
	})
	return i
}

// RetryUntilSuccessful calls callback every interval until duration is exceeded, or until it returns false (to not retry)
func RetryUntilSuccessful(duration time.Duration,
	interval time.Duration,
	callback func() (bool, error)) error {

	wrapFunctionNoResult := func() (interface{}, bool, error) {
		shouldRetry, err := callback()
		return nil, shouldRetry, err
	}

	_, err := RetryUntilSuccessfulWithResult(duration, interval, wrapFunctionNoResult)
	return err
}

// RetryUntilSuccessfulWithResult calls callback every interval until duration is exceeded,
// or until it returns false (to not retry), and returns the result of the callback
func RetryUntilSuccessfulWithResult(duration time.Duration,
	interval time.Duration,
	callback func() (interface{}, bool, error)) (interface{}, error) {
	var (
		lastErr, err error
		result       interface{}
		shouldRetry  bool
	)

	timedOutErrorMessage := "Timed out waiting until successful"
	deadline := time.Now().Add(duration)

	// while we haven't passed the deadline
	for !time.Now().After(deadline) {
		result, shouldRetry, err = callback()
		lastErr = err
		if !shouldRetry {
			return result, err
		}
		time.Sleep(interval)
		continue

	}
	if lastErr != nil {

		// wrap last error
		return result, errors.Wrapf(lastErr, timedOutErrorMessage)
	}

	// duration expired but no error
	return result, lastErr
}
