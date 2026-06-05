//go:build test_unit

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
	"os"
	"path"
	"runtime"
	"strconv"
	"testing"
	"time"

	"github.com/nuclio/errors"
	"github.com/stretchr/testify/suite"
)

type WriteToFileTestSuite struct {
	suite.Suite
}

func (suite *WriteToFileTestSuite) TestWriteToFile() {
	fileName := "test_file.log"
	tmpDir, err := os.MkdirTemp("", "test-*")
	suite.Require().NoError(err)
	filePath := path.Join(tmpDir, fileName)

	// write file
	err = WriteToFile(filePath, []byte("test"), false)
	suite.Require().NoError(err, "Failed to write to file")

	// read file
	fileBytes, err := os.ReadFile(filePath)
	suite.Require().NoError(err, "Failed to read file")

	// verify file content
	suite.Require().Equal("test", string(fileBytes))
}

type EnsureFileExistsTestSuite struct {
	suite.Suite
}

func (suite *EnsureFileExistsTestSuite) TestNoFDLeak() {
	if runtime.GOOS != "linux" {
		suite.T().Skip("FD-count check requires /proc/self/fd (Linux only)")
	}

	tmpDir, err := os.MkdirTemp("", "ensure-noleak-*")
	suite.Require().NoError(err)
	defer os.RemoveAll(tmpDir) // nolint: errcheck

	const iterations = 100
	before := countOpenFDs(suite.T())
	for i := 0; i < iterations; i++ {
		err := EnsureFileExists(path.Join(tmpDir, "f-"+strconv.Itoa(i)))
		suite.Require().NoError(err)
	}
	after := countOpenFDs(suite.T())

	suite.Require().LessOrEqualf(after-before, 10,
		"FD count grew from %d to %d after %d EnsureFileExists calls (expected <=10 growth)",
		before, after, iterations)
}

func countOpenFDs(t *testing.T) int {
	t.Helper()
	entries, err := os.ReadDir("/proc/self/fd")
	if err != nil {
		t.Fatalf("read /proc/self/fd: %v", err)
	}
	return len(entries)
}

type RetryUntilSuccessfulTestSuite struct {
	suite.Suite
}

func (suite *RetryUntilSuccessfulTestSuite) TestNegative() {
	err := RetryUntilSuccessful(50*time.Millisecond, 10*time.Millisecond, func() (bool, error) {
		return false, nil
	})

	suite.Require().NoError(err)
}

func (suite *RetryUntilSuccessfulTestSuite) TestPositive() {
	err := RetryUntilSuccessful(50*time.Millisecond, 10*time.Millisecond, func() (bool, error) {
		return true, errors.New("test")
	})

	suite.Require().Error(err)
}

func (suite *RetryUntilSuccessfulTestSuite) TestRetryWithResult() {
	callCount := 0
	result, err := RetryUntilSuccessfulWithResult(50*time.Millisecond, 10*time.Millisecond, func() (interface{}, bool, error) {
		if callCount == 0 {
			callCount++
			return 0, true, errors.New("test")
		}
		return 1, false, nil
	})
	suite.Require().NoError(err)

	intResult, ok := result.(int)
	suite.Require().True(ok)
	suite.Require().Equal(1, intResult)
}

func TestHelperTestSuite(t *testing.T) {
	suite.Run(t, new(WriteToFileTestSuite))
	suite.Run(t, new(EnsureFileExistsTestSuite))
	suite.Run(t, new(RetryUntilSuccessfulTestSuite))
}
