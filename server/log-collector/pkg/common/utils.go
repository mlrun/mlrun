// Copyright 2023 Iguazio
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
	"bytes"
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

var TimedOutErrorMessage = "Timed out waiting until successful"
var ErrRetryUntilSuccessfulTimeout = errors.New(TimedOutErrorMessage)

// GetEnvOrDefaultString returns the string value of the environment variable with the given key, or the given default
// value if the environment variable is not set
func GetEnvOrDefaultString(key string, defaultValue string) string {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	} else if value == "nil" || value == "none" {
		return ""
	}
	return value
}

// GetEnvOrDefaultBool returns the boolean value of the environment variable with the given key, or the given default
// value if the environment variable is not set
func GetEnvOrDefaultBool(key string, defaultValue bool) bool {
	return strings.ToLower(GetEnvOrDefaultString(key, strconv.FormatBool(defaultValue))) == "true"
}

// GetEnvOrDefaultInt returns the integer value of the environment variable with the given key, or the given default
// value if the environment variable is not set
func GetEnvOrDefaultInt(key string, defaultValue int) int {
	valueInt, err := strconv.Atoi(GetEnvOrDefaultString(key, strconv.Itoa(defaultValue)))
	if err != nil {
		return defaultValue
	}
	return valueInt
}

// GetKubernetesClientConfig returns a kubernetes client config
func GetKubernetesClientConfig(kubeconfigPath string) (*rest.Config, error) {
	if kubeconfigPath != "" {
		return clientcmd.BuildConfigFromFlags("", kubeconfigPath)
	}

	return rest.InClusterConfig()
}

// EnsureDirExists creates a directory if it doesn't exist
func EnsureDirExists(dirPath string, mode os.FileMode) error {
	if _, err := os.Stat(dirPath); os.IsNotExist(err) {
		if err := os.MkdirAll(dirPath, mode); err != nil {
			return errors.Wrap(err, "Failed to create directory")
		}
	}

	return nil
}

// EnsureFileExists creates a file if it doesn't exist
func EnsureFileExists(filePath string) error {
	if exists, err := FileExists(filePath); !exists {
		if err != nil {
			return errors.Wrap(err, "Failed to check if file exists")
		}

		// get file directory
		dirPath := filepath.Dir(filePath)
		if err := EnsureDirExists(dirPath, os.ModePerm); err != nil {
			return errors.Wrapf(err, "Failed to create directory - %s", dirPath)
		}
		if _, err := os.Create(filePath); err != nil {
			return errors.Wrapf(err, "Failed to create file - %s", filePath)
		}
	}

	return nil
}

// FileExists returns true if the given file exists
func FileExists(filePath string) (bool, error) {
	_, err := os.Stat(filePath)
	if err == nil {
		return true, nil

	} else if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}

	// sanity: file may or may not exist
	return false, err
}

// WriteToFile writes the given bytes to the given file path
func WriteToFile(filePath string,
	content []byte,
	append bool) error {

	// this flag enables us to create the file if it doesn't exist, and open the file read/write permissions
	openFlags := os.O_CREATE | os.O_RDWR
	if append {
		openFlags |= os.O_APPEND
	} else {

		// if we're not appending, we want to truncate the file
		openFlags |= os.O_TRUNC
	}

	if err := EnsureFileExists(filePath); err != nil {
		return errors.Wrap(err, "Failed to ensure file exists")
	}

	// open file
	file, err := os.OpenFile(filePath, openFlags, 0600)
	if err != nil {
		return errors.Wrapf(err, "Failed to open file - %s", filePath)
	}

	defer file.Close() // nolint: errcheck

	if _, err := file.Write(content); err != nil {
		return errors.Wrapf(err, "Failed to write log contents to file - %s", filePath)
	}

	return nil
}

// GetFileSize returns the size of the given file
func GetFileSize(filePath string) (int64, error) {
	file, err := os.OpenFile(filePath, os.O_RDONLY, 0600)
	if err != nil {
		return 0, errors.Wrapf(err, "Failed to open log file - %s", filePath)
	}
	defer file.Close() // nolint: errcheck

	fileInfo, err := file.Stat()
	if err != nil {
		return 0, errors.Wrapf(err, "Failed to get file info for file - %s", filePath)
	}

	return fileInfo.Size(), nil
}

// SyncMapLength returns the length of a sync.Map
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

	// do not wrap err, we want to return the original error
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
		return result, errors.Wrapf(lastErr, TimedOutErrorMessage)
	}

	// duration expired, but last callback failed
	if shouldRetry {
		return result, ErrRetryUntilSuccessfulTimeout
	}

	// duration expired, but last callback succeeded
	return result, lastErr
}

func GetErrorStack(err error, depth int) string {
	errorStack := bytes.Buffer{}
	errors.PrintErrorStack(&errorStack, err, depth)
	return errorStack.String()
}
