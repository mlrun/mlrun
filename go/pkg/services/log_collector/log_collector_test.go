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

package log_collector

import (
	"context"
	"os"
	"path"
	"testing"

	"github.com/nuclio/logger"
	"github.com/nuclio/loggerus"
	"github.com/stretchr/testify/suite"
	"k8s.io/client-go/kubernetes/fake"
)

type LogCollectorTestSuite struct {
	suite.Suite
	LogCollectorServer    *LogCollectorServer
	logger                logger.Logger
	ctx                   context.Context
	kubeClientSet         fake.Clientset
	namespace             string
	baseDir               string
	kubeConfigFilePath    string
	monitoringIntervalStr string
	getLogsIntervalStr    string
}

func (suite *LogCollectorTestSuite) SetupSuite() {
	var err error
	suite.logger, err = loggerus.NewLoggerusForTests("test")
	suite.Require().NoError(err, "Failed to create logger")

	suite.kubeClientSet = *fake.NewSimpleClientset()
	suite.ctx = context.Background()
	suite.namespace = "default"
	suite.monitoringIntervalStr = "10s"
	suite.getLogsIntervalStr = "10s"

	// get cwd
	cwd, err := os.Getwd()
	suite.Require().NoError(err, "Failed to get cwd")

	// create base dir
	suite.baseDir = path.Join(cwd, "/log_collector_test")
	err = os.MkdirAll(suite.baseDir, 0777)
	suite.Require().NoError(err, "Failed to create base dir")

	// get kube config file path
	homeDir, err := os.UserHomeDir()
	suite.Require().NoError(err, "Failed to get home dir")
	suite.kubeConfigFilePath = path.Join(homeDir, ".kube", "config")

	// create log collector server
	suite.LogCollectorServer, err = NewLogCollectorServer(suite.logger,
		suite.namespace,
		suite.baseDir,
		suite.kubeConfigFilePath,
		suite.monitoringIntervalStr,
		suite.getLogsIntervalStr)
	suite.Require().NoError(err, "Failed to create log collector server")

	// overwrite log collector server's kube client set with the fake one
	suite.LogCollectorServer.kubeClientSet = &suite.kubeClientSet

	suite.logger.InfoWith("Setup complete")
}

func (suite *LogCollectorTestSuite) TearDownSuite() {

	// delete all files in base dir
	err := os.RemoveAll(suite.baseDir)
	suite.Require().NoError(err, "Failed to delete base dir")

	// delete base dir itself
	err = os.Remove(suite.baseDir)
	suite.Require().NoError(err, "Failed to delete base dir")

	suite.logger.InfoWith("Tear down complete")
}

func (suite *LogCollectorTestSuite) TestValidateOffsetAndSize() {

	for _, testCase := range []struct {
		name           string
		offset         uint64
		size           uint64
		fileSize       uint64
		expectedOffset uint64
		expectedSize   uint64
	}{
		{
			name:           "offset and size are valid",
			offset:         0,
			size:           10,
			fileSize:       100,
			expectedOffset: 0,
			expectedSize:   10,
		},
		{
			name:           "size is larger than file size",
			offset:         0,
			size:           200,
			fileSize:       100,
			expectedOffset: 0,
			expectedSize:   100,
		},
		{
			name:           "size is larger than file size offset difference",
			offset:         20,
			size:           90,
			fileSize:       100,
			expectedOffset: 20,
			expectedSize:   80,
		},
		{
			name:           "size is zero",
			offset:         10,
			size:           0,
			fileSize:       100,
			expectedOffset: 10,
			expectedSize:   100,
		},
		{
			name:           "offset is larger than file size",
			offset:         200,
			size:           50,
			fileSize:       100,
			expectedOffset: 0,
			expectedSize:   50,
		},
	} {
		suite.Run(testCase.name, func() {
			offset, size := suite.LogCollectorServer.validateOffsetAndSize(testCase.offset, testCase.size, testCase.fileSize)
			suite.Require().Equal(testCase.expectedOffset, offset)
			suite.Require().Equal(testCase.expectedSize, size)
		})
	}
}

func (suite *LogCollectorTestSuite) TestWriteToFile() {
	fileName := "test_file.log"
	filePath := path.Join(suite.baseDir, fileName)

	// write file
	err := suite.LogCollectorServer.writeToFile(suite.ctx, filePath, []byte("test"), false)
	suite.Require().NoError(err, "Failed to write to file")

	// read file
	fileBytes, err := os.ReadFile(filePath)
	suite.Require().NoError(err, "Failed to read file")

	// verify file content
	suite.Require().Equal("test", string(fileBytes))
}

func (suite *LogCollectorTestSuite) TestReadWriteStateFile() {

	// read state file
	state, err := suite.LogCollectorServer.getState()
	suite.Require().NoError(err, "Failed to read state file")

	// verify state is empty
	suite.Require().Equal(0, len(state.InProgress))

	// add item to InProgress
	item := LogItem{
		RunId: "abc123",
		LabelSelector: map[string]string{
			"app": "test",
		},
	}

	state.InProgress = append(state.InProgress, item)

	// write state file
	err = suite.LogCollectorServer.writeStateToFile(suite.ctx, state)
	suite.Require().NoError(err, "Failed to write state file")

	// read state file again

	state, err = suite.LogCollectorServer.getState()
	suite.Require().NoError(err, "Failed to read state file")

	// verify item is in progress
	suite.Require().Equal(1, len(state.InProgress))
	suite.Require().Equal(item, state.InProgress[0])
}

func TestLogCollectorTestSuite(t *testing.T) {
	suite.Run(t, new(LogCollectorTestSuite))
}
