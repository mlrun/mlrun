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

package logcollector

import (
	"context"
	"fmt"
	"os"
	"path"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mlrun/mlrun-go/pkg/common"
	"github.com/mlrun/mlrun-go/pkg/proto/build/log_collector"

	"github.com/google/uuid"
	"github.com/mlrun/log-collector/pkg/services/logcollector/test/nop"
	"github.com/nuclio/errors"
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
	logCollectorServer *Server
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
	advancedLogLevel := 0
	listRunsChunkSize := 10

	// create base dir
	suite.baseDir = path.Join(os.TempDir(), "/log_collector_test")
	err = os.MkdirAll(suite.baseDir, 0777)
	suite.Require().NoError(err, "Failed to create base dir")

	// create log collector server
	suite.logCollectorServer, err = NewLogCollectorServer(suite.logger,
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
		bufferSizeBytes,
		common.LogTimeUpdateBytesInterval,
		advancedLogLevel,
		listRunsChunkSize)
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
			offset, size := suite.logCollectorServer.validateOffsetAndSize(testCase.offset, testCase.size, testCase.fileSize)
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

	startedChan := make(chan bool)

	// stream pod logs
	go suite.logCollectorServer.startLogStreaming(context.WithoutCancel(suite.ctx),
		runId,
		pod.Name,
		suite.projectName,
		0,
		startedChan)

	// wait for log streaming to start
	started := <-startedChan
	suite.Require().True(started, "Log streaming didn't start")

	// resolve log file path
	logFilePath := suite.logCollectorServer.resolveRunLogFilePath(suite.projectName, runId)

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
	suite.logCollectorServer.startLogsFindingPodsTimeout = 50 * time.Millisecond
	suite.logCollectorServer.startLogsFindingPodsInterval = 20 * time.Millisecond
	response, err := suite.logCollectorServer.StartLog(suite.ctx, request)
	suite.Require().NoError(err, "Failed to start log")
	suite.Require().True(response.Success, "Failed to start log")
}

func (suite *LogCollectorTestSuite) TestStartLogOnPodStates() {
	selector := "app=some-app"
	projectName := "some-project"
	var runUidIndex int

	// remove project from in-progress cache when test is done
	defer func() {
		err := suite.logCollectorServer.stateManifest.RemoveProject(projectName)
		suite.Require().NoError(err, "Failed to remove project from state manifest")
	}()

	for _, testCase := range []struct {
		name            string
		podPhase        v1.PodPhase
		expectedFailure bool
	}{
		{
			name:            "pod is running",
			podPhase:        v1.PodRunning,
			expectedFailure: false,
		},
		{
			name:            "pod is succeeded",
			podPhase:        v1.PodSucceeded,
			expectedFailure: false,
		},
		{
			name:            "pod is failed",
			podPhase:        v1.PodFailed,
			expectedFailure: false,
		},
		{
			name:            "pod is pending",
			podPhase:        v1.PodPending,
			expectedFailure: true,
		},
	} {
		// not using suite.Run because when the test cases run in parallel the fake client set is shared between them
		// and it causes conflicts
		suite.logger.InfoWith("Running test case", "testName", testCase.name)

		runUidIndex++

		fakePod := v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("test-pod-%d", runUidIndex),
				Labels: map[string]string{
					"app": "some-app",
				},
			},
			Status: v1.PodStatus{
				Phase: testCase.podPhase,
			},
		}

		pod, err := suite.kubeClientSet.CoreV1().Pods(suite.namespace).Create(suite.ctx, &fakePod, metav1.CreateOptions{})
		suite.Require().NoError(err, "Failed to create pod")

		// call start log
		request := &log_collector.StartLogRequest{
			RunUID:      fmt.Sprintf("run-id-%d", runUidIndex),
			ProjectName: projectName,
			Selector:    selector,
		}

		response, err := suite.logCollectorServer.StartLog(suite.ctx, request)
		if testCase.expectedFailure {
			suite.Require().Error(err, "Start log should have failed")
			suite.Require().False(response.Success, "Start log should not have succeeded")
		} else {
			suite.Require().NoError(err, "Start log should not have failed")
			suite.Require().True(response.Success, "Start log should have succeeded")
		}

		// delete pod when test is done
		err = suite.kubeClientSet.CoreV1().Pods(suite.namespace).Delete(suite.ctx, pod.Name, metav1.DeleteOptions{})
		suite.Require().NoError(err, "Failed to delete pod")
	}
}

func (suite *LogCollectorTestSuite) TestGetLogsWithSize() {
	runUID := uuid.New().String()

	// create log file for runUID and pod
	logFilePath := suite.logCollectorServer.resolveRunLogFilePath(suite.projectName, runUID)

	// write log file
	// write 10 times buffer size log lines
	for i := 0; i < 10*suite.logCollectorServer.getLogsBufferSizeBytes; i++ {
		logText := fmt.Sprintf("Some fake pod logs %d\n", i)
		err := common.WriteToFile(logFilePath, []byte(logText), true)
		suite.Require().NoError(err, "Failed to write to file")
	}
	fileContents, err := os.ReadFile(logFilePath)
	suite.Require().NoError(err)

	fileContentsLength := len(fileContents)

	for _, testCase := range []struct {
		name             string
		offset           int
		readSize         int
		expectedReadSize int
		verifyContents   bool
	}{
		{
			name:             "Read it all",
			offset:           0,
			readSize:         -1,
			expectedReadSize: fileContentsLength,
			verifyContents:   true,
		},
		{
			name:             "Read nothing",
			offset:           0,
			readSize:         0,
			expectedReadSize: 0,
		},
		{
			name:             "Read 100 bytes",
			offset:           0,
			readSize:         100,
			expectedReadSize: 100,
			verifyContents:   true,
		},
		{
			name:             "Read buffer size bytes with offset",
			offset:           fileContentsLength / 2,
			readSize:         suite.logCollectorServer.getLogsBufferSizeBytes,
			expectedReadSize: suite.logCollectorServer.getLogsBufferSizeBytes,
			verifyContents:   true,
		},
		{
			name:             "Read buffer size * 2 bytes with offset",
			offset:           fileContentsLength / 2,
			readSize:         suite.logCollectorServer.getLogsBufferSizeBytes * 2,
			expectedReadSize: suite.logCollectorServer.getLogsBufferSizeBytes * 2,
			verifyContents:   true,
		},
		{
			name:             "Read buffer size / 2 bytes with offset",
			offset:           fileContentsLength / 2,
			readSize:         suite.logCollectorServer.getLogsBufferSizeBytes / 2,
			expectedReadSize: suite.logCollectorServer.getLogsBufferSizeBytes / 2,
			verifyContents:   true,
		},
		{

			// further explanation - This edge case is when the offset + readSize + buffer size is larger than the file size
			// so we must reduce the buffer size to fit the needed size
			name:             "Overflowing read size + bufferSize",
			offset:           fileContentsLength - int(1.5*float64(suite.logCollectorServer.getLogsBufferSizeBytes)),
			readSize:         int(1.3 * float64(suite.logCollectorServer.getLogsBufferSizeBytes)),
			expectedReadSize: int(1.3 * float64(suite.logCollectorServer.getLogsBufferSizeBytes)),
			verifyContents:   true,
		},
		{
			name:             "Overflowing read size return what is left",
			offset:           fileContentsLength - 1,
			readSize:         100,
			expectedReadSize: 1,
			verifyContents:   true,
		},
		{
			name:             "Overflowing offset - returns nothing",
			offset:           fileContentsLength,
			readSize:         -1,
			expectedReadSize: 0,
		},
	} {
		suite.Run(testCase.name, func() {
			nopStream := &nop.GetLogsResponseStreamNop{}
			err := suite.logCollectorServer.GetLogs(&log_collector.GetLogsRequest{
				RunUID:      runUID,
				Offset:      int64(testCase.offset),
				Size:        int64(testCase.readSize),
				ProjectName: suite.projectName,
			}, nopStream)
			suite.Require().NoError(err, "Failed to get logs")

			// verify logs
			suite.Require().Equal(testCase.expectedReadSize, len(nopStream.Logs))
			if testCase.verifyContents {
				suite.Require().Equal(string(
					fileContents[testCase.offset:testCase.offset+testCase.expectedReadSize]),
					string(nopStream.Logs))
			}
		})
	}

}

func (suite *LogCollectorTestSuite) TestGetLogsSuccessful() {

	runUID := uuid.New().String()

	// create log file for runUID and pod
	logFilePath := suite.logCollectorServer.resolveRunLogFilePath(suite.projectName, runUID)

	// write log file
	logText := "Some fake pod logs\n"
	err := common.WriteToFile(logFilePath, []byte(logText), false)
	suite.Require().NoError(err, "Failed to write to file")

	// initialize stream
	nopStream := &nop.GetLogsResponseStreamNop{}

	// get logs
	err = suite.logCollectorServer.GetLogs(&log_collector.GetLogsRequest{
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
	err = suite.logCollectorServer.GetLogs(&log_collector.GetLogsRequest{
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

	startReading := make(chan bool)

	var totalWrittenLock sync.Mutex
	var totalWritten = 0
	var totalReadLock sync.Mutex
	var totalRead = 0
	var totalLogLines = 100

	// write to file
	errGroup.Go(func() error {
		for i := 0; i < totalLogLines; i++ {
			if i == 10 {
				startReading <- true
			}
			message := fmt.Sprintf(messageTemplate, i)
			err := common.WriteToFile(filePath, []byte(message), true)
			suite.Require().NoError(err, "Failed to write to file")
			totalWrittenLock.Lock()
			totalWritten += len(message)
			totalWrittenLock.Unlock()
		}
		return nil
	})

	// read from file
	errGroup.Go(func() error {

		// wait before writing to file
		<-startReading
		var offset int
		var readLogs string

		var j int
		for {
			logs, err := suite.logCollectorServer.readLogsFromFile(ctx,
				"1",
				filePath,
				int64(offset),
				int64(len(messageTemplate)))
			if err != nil {
				return err
			}

			offset += len(logs)
			readLogs += string(logs)
			totalReadLock.Lock()
			totalRead += len(logs)
			totalReadLock.Unlock()

			if j == totalLogLines {
				break
			}
			if logs == nil {
				time.Sleep(10 * time.Millisecond)
				suite.logger.DebugWith("Got nil logs, retrying", "offset", offset)
				continue
			}
			j++
		}

		suite.Require().Equal(totalLogLines, strings.Count(readLogs, "\n"), "Expected to read 100 lines")
		return nil
	})

	// wait for goroutines to finish
	suite.Require().NoError(errGroup.Wait(), "Failed to wait for goroutines to finish")
	suite.Require().Equal(totalWritten, totalRead, "Expected total written to be equal to total read")
}

func (suite *LogCollectorTestSuite) TestGetLogSize() {
	runUID := uuid.New().String()
	request := &log_collector.GetLogSizeRequest{
		RunUID:      runUID,
		ProjectName: suite.projectName,
	}

	// call get log size with no logs
	getLogSizeResponse, err := suite.logCollectorServer.GetLogSize(suite.ctx, request)
	suite.Require().NoError(err, "Failed to get log size")
	suite.Require().True(getLogSizeResponse.Success, "Expected get log size request to succeed")
	suite.Require().Equal(int64(-1), getLogSizeResponse.LogSize, "Expected size to be negative")

	// create log file for runUID and pod
	logFilePath := suite.logCollectorServer.resolveRunLogFilePath(suite.projectName, runUID)

	// write log file
	logText := "Some fake pod logs\n"
	err = common.WriteToFile(logFilePath, []byte(logText), false)
	suite.Require().NoError(err, "Failed to write to file")

	// get the log size
	getLogSizeResponse, err = suite.logCollectorServer.GetLogSize(suite.ctx, request)
	suite.Require().NoError(err, "Failed to get log size")
	suite.Require().True(getLogSizeResponse.Success, "Expected get log size request to succeed")
	suite.Require().Equal(int64(len(logText)), getLogSizeResponse.LogSize, "Expected size to be equal to log size")
}

func (suite *LogCollectorTestSuite) TestStopLog() {
	var err error

	logItemsNum := 5
	projectNum := 2
	var projectToRuns = map[string][]string{}

	suite.createLogItems(projectNum, logItemsNum, projectToRuns)

	// write state
	err = suite.logCollectorServer.stateManifest.WriteState(suite.logCollectorServer.stateManifest.GetState())
	suite.Require().NoError(err, "Failed to write state")

	// verify all items are in progress
	logItemsInProgress, err := suite.logCollectorServer.stateManifest.GetItemsInProgress()
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
		request := &log_collector.StopLogsRequest{
			Project: project,
			RunUIDs: runs,
		}
		response, err := suite.logCollectorServer.StopLogs(suite.ctx, request)
		suite.Require().NoError(err, "Failed to stop log")
		suite.Require().True(response.Success, "Expected stop log request to succeed")
	}

	// write state again
	err = suite.logCollectorServer.stateManifest.WriteState(suite.logCollectorServer.stateManifest.GetState())
	suite.Require().NoError(err, "Failed to write state")

	// verify no items in progress
	logItemsInProgress, err = suite.logCollectorServer.stateManifest.GetItemsInProgress()
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
				logFilePath := suite.logCollectorServer.resolveRunLogFilePath(projectName, runUID)
				err := common.WriteToFile(logFilePath, []byte("some log"), false)
				suite.Require().NoError(err, "Failed to write to file")
			}

			// verify files exist
			dirPath := path.Join(suite.logCollectorServer.baseDir, projectName)
			dirEntries, err := os.ReadDir(dirPath)
			suite.Require().NoError(err, "Failed to read dir")
			suite.Require().Equal(testCase.logsNumToCreate, len(dirEntries), "Expected logs to exist")

			// delete some logs
			request := &log_collector.StopLogsRequest{
				Project: projectName,
				RunUIDs: runUIDs[testCase.expectedLogsNumLeft:],
			}
			response, err := suite.logCollectorServer.DeleteLogs(suite.ctx, request)
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
		logFilePath := suite.logCollectorServer.resolveRunLogFilePath(projectName, runUID)
		err := common.WriteToFile(logFilePath, []byte("some log"), false)
		suite.Require().NoError(err, "Failed to write to file")
	}

	// verify files exist
	dirPath := path.Join(suite.logCollectorServer.baseDir, projectName)
	dirEntries, err := os.ReadDir(dirPath)
	suite.Require().NoError(err, "Failed to read dir")
	suite.Require().Equal(logsNum, len(dirEntries), "Expected logs to exist")

	// delete all logs except the first one
	request := &log_collector.StopLogsRequest{
		Project: projectName,
		RunUIDs: runUIDs[1:],
	}
	response, err := suite.logCollectorServer.DeleteLogs(suite.ctx, request)
	suite.Require().NoError(err, "Failed to stop log")
	suite.Require().True(response.Success, "Expected stop log request to succeed")

	// verify files deleted
	dirEntries, err = os.ReadDir(dirPath)
	suite.Require().NoError(err, "Failed to read dir")
	suite.Require().Equal(1, len(dirEntries), "Expected logs to be deleted")
}

func (suite *LogCollectorTestSuite) TestGetLogFilePath() {
	runUID := "123"
	projectName := "someProject"
	_, err := suite.logCollectorServer.getLogFilePath(suite.ctx, runUID, projectName)
	suite.Require().Error(err, "Expected error when getting log file path for non-existing project")
	suite.Require().Contains(errors.RootCause(err).Error(), "not found", "Expected error to contain 'not found'")

	// make the project dir
	err = os.MkdirAll(path.Join(suite.baseDir, projectName), 0755)
	suite.Require().NoError(err)

	// make the run file
	runFilePath := suite.logCollectorServer.resolveRunLogFilePath(projectName, runUID)
	err = common.WriteToFile(runFilePath, []byte("some log"), false)
	suite.Require().NoError(err, "Failed to write to file")

	// get the log file path
	logFilePath, err := suite.logCollectorServer.getLogFilePath(suite.ctx, runUID, projectName)
	suite.Require().NoError(err, "Failed to get log file path")
	suite.Require().Equal(runFilePath, logFilePath, "Expected log file path to be the same as the run file path")
}

func (suite *LogCollectorTestSuite) TestListRunsInProgress() {
	listRunsInProgress := func(request *log_collector.ListRunsRequest) []string {
		nopStream := &nop.ListRunsResponseStreamNop{}
		err := suite.logCollectorServer.ListRunsInProgress(request, nopStream)
		suite.Require().NoError(err, "Failed to list runs in progress")
		return nopStream.RunUIDs

	}

	verifyRuns := func(expectedRunUIDs []string, responseRunUIDs []string) {
		suite.Require().Equal(len(expectedRunUIDs), len(responseRunUIDs))
		for _, runUID := range responseRunUIDs {
			suite.Require().Contains(expectedRunUIDs, runUID, "Expected runUID to be in the expected list")
		}
	}

	// list runs without any runs in progress
	runsInProgress := listRunsInProgress(&log_collector.ListRunsRequest{})
	suite.Require().Empty(runsInProgress, "Expected no runs in progress")

	// create log items in progress
	projectNum := 5
	logItemsNum := 5
	var projectToRuns = map[string][]string{}
	suite.createLogItems(projectNum, logItemsNum, projectToRuns)
	defer func() {
		// remove projects from state manifest and current state when test is done to avoid conflicts with other tests
		for project := range projectToRuns {
			err := suite.logCollectorServer.stateManifest.RemoveProject(project)
			suite.Assert().NoError(err, "Failed to remove project from state manifest")

			err = suite.logCollectorServer.currentState.RemoveProject(project)
			suite.Assert().NoError(err, "Failed to remove project from current state")
		}
	}()

	var expectedRunUIDs []string
	for _, runs := range projectToRuns {
		expectedRunUIDs = append(expectedRunUIDs, runs...)
	}

	// list runs in progress for all projects
	runsInProgress = listRunsInProgress(&log_collector.ListRunsRequest{})
	verifyRuns(expectedRunUIDs, runsInProgress)

	// list runs in progress for a specific project
	projectName := "project-1"
	expectedRunUIDs = projectToRuns[projectName]
	runsInProgress = listRunsInProgress(&log_collector.ListRunsRequest{Project: projectName})
	verifyRuns(expectedRunUIDs, runsInProgress)
}

// createLogItems creates `logItemsNum` log items for `projectNum` projects, and adds them to the server's states
func (suite *LogCollectorTestSuite) createLogItems(projectNum int, logItemsNum int, projectToRuns map[string][]string) {
	var err error
	for i := 0; i < projectNum; i++ {
		projectName := fmt.Sprintf("project-%d", i)

		// add log item to the server's states, so no error will be returned
		for j := 0; j < logItemsNum; j++ {
			runUID := uuid.New().String()
			projectToRuns[projectName] = append(projectToRuns[projectName], runUID)
			selector := fmt.Sprintf("run=%s", runUID)

			// Add state to the log collector's state manifest
			err = suite.logCollectorServer.stateManifest.AddLogItem(suite.ctx, runUID, selector, projectName)
			suite.Require().NoError(err, "Failed to add log item to the state manifest")

			// Add state to the log collector's current state
			err = suite.logCollectorServer.currentState.AddLogItem(suite.ctx, runUID, selector, projectName)
			suite.Require().NoError(err, "Failed to add log item to the current state")
		}
	}
}

func TestLogCollectorTestSuite(t *testing.T) {
	suite.Run(t, new(LogCollectorTestSuite))
}
