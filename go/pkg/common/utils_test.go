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
	"path"
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

func TestHelperTestSuite(t *testing.T) {
	suite.Run(t, new(WriteToFileTestSuite))
	suite.Run(t, new(RetryUntilSuccessfulTestSuite))
}
