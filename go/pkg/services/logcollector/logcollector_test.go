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

package logcollector

import (
	"context"
	"fmt"
	"os"
	"path"
	"sync"
	"testing"
	"time"

	"github.com/mlrun/mlrun/pkg/common"
	"github.com/mlrun/mlrun/proto/build/log_collector"

	"github.com/nuclio/logger"
	"github.com/nuclio/loggerus"
	"github.com/stretchr/testify/suite"
	"golang.org/x/sync/errgroup"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

type LogCollectorTestSuite struct {
	suite.Suite
	LogCollectorServer *LogCollectorServer
	logger             logger.Logger
	ctx                context.Context
	kubeClientSet      fake.Clientset
	namespace          string
	baseDir            string
	kubeConfigFilePath string
	fileLock           sync.Locker
}

func (suite *LogCollectorTestSuite) SetupSuite() {
	var err error
	suite.logger, err = loggerus.NewLoggerusForTests("test")
	suite.Require().NoError(err, "Failed to create logger")

	suite.kubeClientSet = *fake.NewSimpleClientset()
	suite.ctx = context.Background()
	suite.namespace = "default"
	stateFileUpdateIntervalStr := "5s"
	readLogWaitTime := "3s"
	bufferSizeBytes := 512
	suite.fileLock = &sync.Mutex{}

	// create base dir
	suite.baseDir = path.Join(os.TempDir(), "/log_collector_test")
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
		stateFileUpdateIntervalStr,
		readLogWaitTime,
		bufferSizeBytes)
	suite.Require().NoError(err, "Failed to create log collector server")

	// overwrite log collector server's kube client set with the fake one
	suite.LogCollectorServer.kubeClientSet = &suite.kubeClientSet

	suite.logger.InfoWith("Setup complete")
}

func (suite *LogCollectorTestSuite) TearDownSuite() {

	// delete base dir and created files
	err := os.RemoveAll(suite.baseDir)
	suite.Require().NoError(err, "Failed to delete base dir")

	suite.logger.InfoWith("Tear down complete")
}

func (suite *LogCollectorTestSuite) TestValidateOffsetAndSize() {

	for _, testCase := range []struct {
		name           string
		offset         uint64
		size           int64
		fileSize       int64
		expectedOffset uint64
		expectedSize   int64
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
			expectedSize:   90,
		},
		{
			name:           "offset is larger than file size",
			offset:         200,
			size:           50,
			fileSize:       100,
			expectedOffset: 0,
			expectedSize:   50,
		},
		{
			name:           "size is negative",
			offset:         50,
			size:           -1,
			fileSize:       100,
			expectedOffset: 50,
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
	err := common.WriteToFile(suite.ctx, suite.logger, filePath, []byte("test"), false)
	suite.Require().NoError(err, "Failed to write to file")

	// read file
	fileBytes, err := os.ReadFile(filePath)
	suite.Require().NoError(err, "Failed to read file")

	// verify file content
	suite.Require().Equal("test", string(fileBytes))
}

func (suite *LogCollectorTestSuite) TestReadWriteStateFile() {

	// read state file
	logItemsInProgress, err := suite.LogCollectorServer.stateStore.GetInProgress()
	suite.Require().NoError(err, "Failed to read state file")

	// verify no items in progress
	suite.Require().Equal(0, len(logItemsInProgress))

	// add a log item to the state file
	runId := "abc123"
	item := LogItem{
		RunUID:        runId,
		LabelSelector: "app=test",
	}

	logItemsInProgress[runId] = item

	// write state file
	err = suite.LogCollectorServer.stateStore.WriteState(&State{
		logItemsInProgress,
	})
	suite.Require().NoError(err, "Failed to write state file")

	// read state file again
	logItemsInProgress, err = suite.LogCollectorServer.stateStore.GetInProgress()
	suite.Require().NoError(err, "Failed to read state file")

	// verify item is in progress
	suite.Require().Equal(1, len(logItemsInProgress))
	suite.Require().Equal(item, logItemsInProgress[runId])
}

func (suite *LogCollectorTestSuite) TestAddRemoveItemFromInProgress() {
	runId := "some-run-id"
	labelSelector := "app=test"

	err := suite.LogCollectorServer.stateStore.AddLogItem(suite.ctx, runId, labelSelector)
	suite.Require().NoError(err, "Failed to add item to in progress")

	// write state to file
	err = suite.LogCollectorServer.stateStore.WriteState(suite.LogCollectorServer.stateStore.GetState())
	suite.Require().NoError(err, "Failed to write state file")

	// read state file
	itemsInProgress, err := suite.LogCollectorServer.stateStore.GetInProgress()
	suite.Require().NoError(err, "Failed to read state file")

	// verify item is in progress
	suite.Require().Equal(1, len(itemsInProgress))
	suite.Require().Equal(runId, itemsInProgress[runId].RunUID)
	suite.Require().Equal(labelSelector, itemsInProgress[runId].LabelSelector)

	// remove item from in progress
	err = suite.LogCollectorServer.stateStore.RemoveLogItem(runId)
	suite.Require().NoError(err, "Failed to remove item from in progress")

	// write state to file again
	err = suite.LogCollectorServer.stateStore.WriteState(suite.LogCollectorServer.stateStore.GetState())
	suite.Require().NoError(err, "Failed to write state file")

	// read state file again
	itemsInProgress, err = suite.LogCollectorServer.stateStore.GetInProgress()
	suite.Require().NoError(err, "Failed to read state file")

	// verify item is not in progress
	suite.Require().Equal(0, len(itemsInProgress))
}

func (suite *LogCollectorTestSuite) TestStreamPodLogs() {
	runId := "some-run-id"
	suite.logger.Debug("Starting test TOMERRRRR")

	// create fake pod that finishes after 10 seconds
	fakePod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: suite.namespace,
			Labels: map[string]string{
				"app": "test",
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "test-container",
					Image: "alpine",
					Command: []string{
						"/bin/sh",
					},
					Args: []string{
						"-c",
						"echo test; sleep 10",
					},
				},
			},
		},
	}

	// add fake pod to fake client set
	pod, err := suite.kubeClientSet.CoreV1().Pods(suite.namespace).Create(suite.ctx, &fakePod, metav1.CreateOptions{})
	suite.Require().NoError(err, "Failed to create pod")

	// stream pod logs
	go suite.LogCollectorServer.startLogStreaming(suite.ctx, runId, pod.Name)

	// resolve log file path
	logFilePath := suite.LogCollectorServer.resolvePodLogFilePath(runId, pod.Name)

	// read log file until it has content, or timeout
	timeout := time.After(30 * time.Second)
	var logFileContent []byte
	foundLogs := false
	for {
		if foundLogs {
			break
		}
		select {
		case <-timeout:
			suite.Require().Fail("Timed out waiting for log file to have content")
		case <-time.Tick(1 * time.Second):

			// read log file
			logFileContent, err = os.ReadFile(logFilePath)
			if err != nil {
				continue
			}
			foundLogs = true
			break
		}
	}

	// verify log file content
	suite.Require().Equal("fake logs", string(logFileContent))
}

func (suite *LogCollectorTestSuite) TestGetLog() {

	runId := "some-run-id"
	podName := "my-pod"

	// creat log file for runId and pod
	logFilePath := suite.LogCollectorServer.resolvePodLogFilePath(runId, podName)

	// write log file
	logText := "Some fake pod logs"
	err := common.WriteToFile(suite.ctx, suite.logger, logFilePath, []byte(logText), false)
	suite.Require().NoError(err, "Failed to write to file")

	// get logs
	log, err := suite.LogCollectorServer.GetLogs(suite.ctx, &log_collector.GetLogsRequest{
		RunUID: runId,
		Offset: 0,
		Size:   100,
	})
	suite.Require().NoError(err, "Failed to get logs")

	// verify logs
	suite.Require().Equal(logText, string(log.Logs))
}

func (suite *LogCollectorTestSuite) TestReadWriteFileSimultaneously() {

	// create file
	filePath := path.Join(suite.baseDir, "test-file")
	suite.logger.DebugWith("Creating file", "path", filePath)
	err := common.EnsureFileExists(filePath)
	suite.Require().NoError(err, "Failed to create file")

	messageTemplate := "This is a test message %d\n"

	errGroup, ctx := errgroup.WithContext(suite.ctx)

	startedWriting := make(chan bool)

	// write to file
	errGroup.Go(func() error {
		signaled := false
		for i := 0; i < 100; i++ {
			if i > 5 && !signaled {
				startedWriting <- true
				signaled = true
			}

			// sleep for a bit to let the other goroutine read from the file
			if i%10 == 0 {
				time.Sleep(1 * time.Second)
			}
			message := fmt.Sprintf(messageTemplate, i)
			suite.logger.DebugWith("Writing to file", "message", message)

			err := common.WriteToFile(ctx, suite.logger, filePath, []byte(message), true)
			suite.Require().NoError(err, "Failed to write to file")
		}
		return nil
	})

	// read from file
	errGroup.Go(func() error {

		// let some logs be written
		<-startedWriting
		time.Sleep(500 * time.Millisecond)

		offset := 0
		for j := 0; j < 100; j++ {

			// sleep for a bit to let the other goroutine write to the file
			if j%10 == 0 {
				time.Sleep(1 * time.Second)
			}

			message := fmt.Sprintf(messageTemplate, j)
			size := int64(len(message))
			logs, err := suite.LogCollectorServer.readLogsFromFile(ctx, "1", filePath, uint64(offset), size)
			suite.Require().NoError(err, "Failed to read logs from file")

			// verify logs
			suite.logger.DebugWith("Read from file", "offset", offset, "logs", string(logs))
			suite.Require().Equal(message, string(logs))
			offset += len(message)
		}

		return nil
	})

	// wait for goroutines to finish
	err = errGroup.Wait()
	suite.Require().NoError(err, "Failed to wait for goroutines to finish")
}

func TestLogCollectorTestSuite(t *testing.T) {
	suite.Run(t, new(LogCollectorTestSuite))
}
