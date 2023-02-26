//go:build test_unit

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
	"github.com/mlrun/mlrun/pkg/services/logcollector/test/nop"
	"github.com/mlrun/mlrun/proto/build/log_collector"

	"github.com/google/uuid"
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
	LogCollectorServer *Server
	logger             logger.Logger
	ctx                context.Context
	kubeClientSet      fake.Clientset
	namespace          string
	projectName        string
	baseDir            string
}

func (suite *LogCollectorTestSuite) SetupSuite() {
	var err error
	suite.logger, err = loggerus.NewLoggerusForTests("test")
	suite.Require().NoError(err, "Failed to create logger")

	suite.kubeClientSet = *fake.NewSimpleClientset()
	suite.ctx = context.Background()
	suite.namespace = "default"
	suite.projectName = "test-project"
	stateFileUpdateIntervalStr := "5s"
	readLogWaitTime := "3s"
	monitoringInterval := "30s"
	bufferPoolSize := 20
	bufferSizeBytes := 100
	clusterizationRole := "chief"

	// create base dir
	suite.baseDir = path.Join(os.TempDir(), "/log_collector_test")
	err = os.MkdirAll(suite.baseDir, 0777)
	suite.Require().NoError(err, "Failed to create base dir")

	// create log collector server
	suite.LogCollectorServer, err = NewLogCollectorServer(suite.logger,
		suite.namespace,
		suite.baseDir,
		stateFileUpdateIntervalStr,
		readLogWaitTime,
		monitoringInterval,
		clusterizationRole,
		&suite.kubeClientSet,
		bufferPoolSize,
		bufferPoolSize,
		bufferSizeBytes,
		bufferSizeBytes)
	suite.Require().NoError(err, "Failed to create log collector server")

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
		offset         int64
		size           int64
		fileSize       int64
		expectedOffset int64
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
			expectedOffset: 200,
			expectedSize:   0,
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

func (suite *LogCollectorTestSuite) TestStreamPodLogs() {
	runId := "some-run-id"

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

	ctx, cancel := context.WithCancel(suite.ctx)
	startedChan := make(chan bool)

	// stream pod logs
	go suite.LogCollectorServer.startLogStreaming(ctx, runId, pod.Name, suite.projectName, startedChan, cancel)

	// wait for log streaming to start
	started := <-startedChan
	suite.Require().True(started, "Log streaming didn't start")

	// resolve log file path
	logFilePath := suite.LogCollectorServer.resolvePodLogFilePath(suite.projectName, runId, pod.Name)

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
			if err != nil || len(logFileContent) == 0 {
				continue
			}
			foundLogs = true
			break
		}
	}

	// verify log file content
	suite.Require().Contains(string(logFileContent), "fake logs")
}

func (suite *LogCollectorTestSuite) TestStartLogBestEffort() {

	// call start log for a non-existent pod, and expect no error
	request := &log_collector.StartLogRequest{
		RunUID:      "some-run-id",
		ProjectName: "some-project",
		Selector:    "app=some-app",
		BestEffort:  true,
	}
	response, err := suite.LogCollectorServer.StartLog(suite.ctx, request)
	suite.Require().NoError(err, "Failed to start log")
	suite.Require().True(response.Success, "Failed to start log")
}

func (suite *LogCollectorTestSuite) TestGetLogsSuccessful() {

	runUID := uuid.New().String()
	podName := "my-pod"

	// creat log file for runUID and pod
	logFilePath := suite.LogCollectorServer.resolvePodLogFilePath(suite.projectName, runUID, podName)

	// write log file
	logText := "Some fake pod logs\n"
	err := common.WriteToFile(logFilePath, []byte(logText), false)
	suite.Require().NoError(err, "Failed to write to file")

	// initialize stream
	nopStream := &nop.GetLogsResponseStreamNop{}

	// get logs
	err = suite.LogCollectorServer.GetLogs(&log_collector.GetLogsRequest{
		RunUID:      runUID,
		Offset:      0,
		Size:        100,
		ProjectName: suite.projectName,
	}, nopStream)
	suite.Require().NoError(err, "Failed to get logs")

	// verify logs
	suite.Require().Equal(logText, string(nopStream.Logs))

	// clean mock stream logs
	nopStream.Logs = []byte{}

	expectedResultLogs := []byte(logText)

	suite.logger.DebugWith("Writing more logs to file", "logFilePath", logFilePath)

	// write more logs to log file
	for i := 0; i < 5; i++ {
		expectedResultLogs = append(expectedResultLogs, []byte(logText)...)
		err = common.WriteToFile(logFilePath, []byte(logText), true)
		suite.Require().NoError(err, "Failed to write to file")
	}

	// get logs with offset and size -1, to get all logs at once
	err = suite.LogCollectorServer.GetLogs(&log_collector.GetLogsRequest{
		RunUID:      runUID,
		Offset:      1,
		Size:        -1,
		ProjectName: suite.projectName,
	}, nopStream)
	suite.Require().NoError(err, "Failed to get logs")

	// truncate expected logs to offset
	expectedResultLogs = expectedResultLogs[1:]

	// verify logs
	suite.Require().Equal(expectedResultLogs, nopStream.Logs)
}

func (suite *LogCollectorTestSuite) TestReadLogsFromFileWhileWriting() {

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

			err := common.WriteToFile(filePath, []byte(message), true)
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
			logs, err := suite.LogCollectorServer.readLogsFromFile(ctx, "1", filePath, int64(offset), size)
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

func (suite *LogCollectorTestSuite) TestHasLogs() {
	runUID := uuid.New().String()
	podName := "my-pod"

	request := &log_collector.HasLogsRequest{
		RunUID:      runUID,
		ProjectName: suite.projectName,
	}

	// call has logs with no logs
	hasLogsResponse, err := suite.LogCollectorServer.HasLogs(suite.ctx, request)
	suite.Require().NoError(err, "Failed to check if has logs")
	suite.Require().True(hasLogsResponse.Success, "Expected has logs request to succeed")
	suite.Require().False(hasLogsResponse.HasLogs, "Expected run to not have logs")

	// create log file for runUID and pod
	logFilePath := suite.LogCollectorServer.resolvePodLogFilePath(suite.projectName, runUID, podName)

	// write log file
	logText := "Some fake pod logs\n"
	err = common.WriteToFile(logFilePath, []byte(logText), false)
	suite.Require().NoError(err, "Failed to write to file")

	// check if run has logs
	hasLogsResponse, err = suite.LogCollectorServer.HasLogs(suite.ctx, request)
	suite.Require().NoError(err, "Failed to check if has logs")
	suite.Require().True(hasLogsResponse.Success, "Expected has logs request to succeed")
	suite.Require().True(hasLogsResponse.HasLogs, "Expected run to have logs")
}

func (suite *LogCollectorTestSuite) TestStopLog() {
	var err error

	logItemsNum := 5
	projectNum := 2
	var projectToRuns = map[string][]string{}

	for i := 0; i < projectNum; i++ {
		projectName := fmt.Sprintf("project-%d", i)

		// add log item to the server's states, so no error will be returned
		for j := 0; j < logItemsNum; j++ {
			runUID := uuid.New().String()
			projectToRuns[projectName] = append(projectToRuns[projectName], runUID)
			selector := fmt.Sprintf("run=%s", runUID)

			// Add state to the log collector's state manifest
			err = suite.LogCollectorServer.stateManifest.AddLogItem(suite.ctx, runUID, selector, projectName)
			suite.Require().NoError(err, "Failed to add log item to the state manifest")

			// Add state to the log collector's current state
			err = suite.LogCollectorServer.currentState.AddLogItem(suite.ctx, runUID, selector, projectName)
			suite.Require().NoError(err, "Failed to add log item to the current state")
		}
	}

	// write state
	err = suite.LogCollectorServer.stateManifest.WriteState(suite.LogCollectorServer.stateManifest.GetState())
	suite.Require().NoError(err, "Failed to write state")

	// verify all items are in progress
	logItemsInProgress, err := suite.LogCollectorServer.stateManifest.GetItemsInProgress()
	suite.Require().NoError(err, "Failed to get items in progress")

	suite.Require().Equal(projectNum, common.SyncMapLength(logItemsInProgress), "Expected items to be in progress")
	logItemsInProgress.Range(func(key, value interface{}) bool {
		runUIDsInProgress := value.(*sync.Map)
		suite.Require().Equal(logItemsNum,
			common.SyncMapLength(runUIDsInProgress),
			"Expected items to be in progress")
		return true
	})

	// stop logs for all projects
	for project, runs := range projectToRuns {
		request := &log_collector.StopLogRequest{
			Project: project,
			RunUIDs: runs,
		}
		response, err := suite.LogCollectorServer.StopLog(suite.ctx, request)
		suite.Require().NoError(err, "Failed to stop log")
		suite.Require().True(response.Success, "Expected stop log request to succeed")
	}

	// write state again
	err = suite.LogCollectorServer.stateManifest.WriteState(suite.LogCollectorServer.stateManifest.GetState())
	suite.Require().NoError(err, "Failed to write state")

	// verify no items in progress
	logItemsInProgress, err = suite.LogCollectorServer.stateManifest.GetItemsInProgress()
	suite.Require().NoError(err, "Failed to get items in progress")

	suite.Require().Equal(0,
		common.SyncMapLength(logItemsInProgress),
		"Expected no items in progress")
}

func (suite *LogCollectorTestSuite) TestDeleteLogs() {

	projectCount := 0

	for _, testCase := range []struct {
		name                string
		logsNumToCreate     int
		expectedLogsNumLeft int
	}{
		{
			name:                "Delete some logs",
			logsNumToCreate:     5,
			expectedLogsNumLeft: 2,
		},
		{
			name:                "Delete all logs",
			logsNumToCreate:     5,
			expectedLogsNumLeft: 0,
		},
	} {
		suite.Run(testCase.name, func() {

			// create some log files
			projectName := fmt.Sprintf("test-project-%d", projectCount)
			projectCount++
			var runUIDs []string
			for i := 0; i < testCase.logsNumToCreate; i++ {
				runUID := uuid.New().String()
				runUIDs = append(runUIDs, runUID)
				logFilePath := suite.LogCollectorServer.resolvePodLogFilePath(projectName, runUID, "pod")
				err := common.WriteToFile(logFilePath, []byte("some log"), false)
				suite.Require().NoError(err, "Failed to write to file")
			}

			// verify files exist
			dirPath := path.Join(suite.LogCollectorServer.baseDir, projectName)
			dirEntries, err := os.ReadDir(dirPath)
			suite.Require().NoError(err, "Failed to read dir")
			suite.Require().Equal(testCase.logsNumToCreate, len(dirEntries), "Expected logs to exist")

			// delete all logs except the first one
			request := &log_collector.StopLogRequest{
				Project: projectName,
				RunUIDs: runUIDs[testCase.expectedLogsNumLeft:],
			}
			response, err := suite.LogCollectorServer.DeleteLogs(suite.ctx, request)
			suite.Require().NoError(err, "Failed to stop log")
			suite.Require().True(response.Success, "Expected stop log request to succeed")

			// verify files deleted
			dirEntries, err = os.ReadDir(dirPath)
			suite.Require().NoError(err, "Failed to read dir")
			suite.Require().Equal(testCase.expectedLogsNumLeft, len(dirEntries), "Expected logs to be deleted")
		})
	}
}

func (suite *LogCollectorTestSuite) TestDeleteProjectLogs() {

	// create some log files
	projectName := "test-project"
	logsNum := 5
	var runUIDs []string
	for i := 0; i < logsNum; i++ {
		runUID := uuid.New().String()
		runUIDs = append(runUIDs, runUID)
		logFilePath := suite.LogCollectorServer.resolvePodLogFilePath(projectName, runUID, "pod")
		err := common.WriteToFile(logFilePath, []byte("some log"), false)
		suite.Require().NoError(err, "Failed to write to file")
	}

	// verify files exist
	dirPath := path.Join(suite.LogCollectorServer.baseDir, projectName)
	dirEntries, err := os.ReadDir(dirPath)
	suite.Require().NoError(err, "Failed to read dir")
	suite.Require().Equal(logsNum, len(dirEntries), "Expected logs to exist")

	// delete all logs except the first one
	request := &log_collector.StopLogRequest{
		Project: projectName,
		RunUIDs: runUIDs[1:],
	}
	response, err := suite.LogCollectorServer.DeleteLogs(suite.ctx, request)
	suite.Require().NoError(err, "Failed to stop log")
	suite.Require().True(response.Success, "Expected stop log request to succeed")

	// verify files deleted
	dirEntries, err = os.ReadDir(dirPath)
	suite.Require().NoError(err, "Failed to read dir")
	suite.Require().Equal(1, len(dirEntries), "Expected logs to be deleted")
}

func TestLogCollectorTestSuite(t *testing.T) {
	suite.Run(t, new(LogCollectorTestSuite))
}
