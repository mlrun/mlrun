//go:build test_integration

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

package test

import (
	"context"
	"fmt"
	"io"
	"os"
	"path"
	"strings"
	"testing"
	"time"

	"github.com/mlrun/mlrun/pkg/common"
	"github.com/mlrun/mlrun/pkg/framework"
	"github.com/mlrun/mlrun/pkg/services/logcollector"
	"github.com/mlrun/mlrun/pkg/services/logcollector/test/nop"
	"github.com/mlrun/mlrun/proto/build/log_collector"

	"github.com/google/uuid"
	"github.com/nuclio/logger"
	"github.com/nuclio/loggerus"
	"github.com/stretchr/testify/suite"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

type LogCollectorTestSuite struct {
	suite.Suite
	LogCollectorServer *logcollector.Server
	logger             logger.Logger
	ctx                context.Context
	kubeClientSet      kubernetes.Interface
	namespace          string
	baseDir            string
	bufferSizeBytes    int
}

func (suite *LogCollectorTestSuite) SetupSuite() {
	var err error
	suite.logger, err = loggerus.NewLoggerusForTests("logcollector-test")
	suite.Require().NoError(err, "Failed to create logger")

	// create base dir
	suite.baseDir = path.Join(os.TempDir(), "/log_collector_test")
	err = os.MkdirAll(suite.baseDir, 0777)
	suite.Require().NoError(err, "Failed to create base dir")

	// get kube config file path
	homeDir, err := os.UserHomeDir()
	suite.Require().NoError(err, "Failed to get home dir")
	kubeConfigFilePath := common.GetEnvOrDefaultString("KUBECONFIG", path.Join(homeDir, ".kube", "config"))

	restConfig, err := common.GetKubernetesClientConfig(kubeConfigFilePath)
	suite.Require().NoError(err)

	suite.kubeClientSet, err = kubernetes.NewForConfig(restConfig)
	suite.Require().NoError(err)

	suite.ctx = context.Background()
	suite.namespace = "mlrun-integ-test"
	suite.bufferSizeBytes = 1024

	_, err = suite.kubeClientSet.CoreV1().Namespaces().Create(context.Background(), &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: suite.namespace,
		},
	}, metav1.CreateOptions{})
	suite.Require().NoError(err, "Failed to create namespace")

	// TODO: move to test setup / teardown
	suite.LogCollectorServer, err = logcollector.NewLogCollectorServer(suite.logger,
		suite.namespace,
		suite.baseDir,
		"5s",    /* stateFileUpdateIntervalStr */
		"3s",    /* readLogWaitTime */
		"30s",   /* monitoringInterval */
		"chief", /* clusterizationRole */
		suite.kubeClientSet,
		30,                    /* logCollectionBufferPoolSize */
		30,                    /* getLogsBufferSizeBytes */
		suite.bufferSizeBytes, /* logCollectionBufferSizeBytes */
		suite.bufferSizeBytes, /* getLogsBufferSizeBytes */
		common.LogTimeUpdateBytesInterval)
	suite.Require().NoError(err, "Failed to create log collector server")

	// start log collector server in a goroutine, so it won't block the test
	go suite.startLogCollectorServer(
		8282, /* listenPort */
	)

	suite.logger.InfoWith("Setup completed")
}

func (suite *LogCollectorTestSuite) SetupTest() {
	suite.logger.InfoWith("Running test", "testName", suite.T().Name())
}

func (suite *LogCollectorTestSuite) TearDownSuite() {

	// delete namespace
	err := suite.kubeClientSet.CoreV1().Namespaces().Delete(context.Background(), suite.namespace, metav1.DeleteOptions{})
	suite.Require().NoError(err, "Failed to delete namespace")

	// delete base dir and created files
	err = os.RemoveAll(suite.baseDir)
	suite.Require().NoError(err, "Failed to delete base dir")
	suite.logger.InfoWith("Tear down complete", "testName", suite.T().Name())
}

func (suite *LogCollectorTestSuite) TestLogCollector() {

	// create pod that prints logs
	expectedLogLines := 100
	podName := "test-pod"
	projectName := "test-project"
	runUID := uuid.New().String()
	pod := suite.getDummyPodSpec(podName, expectedLogLines)

	_, err := suite.kubeClientSet.CoreV1().Pods(suite.namespace).Create(suite.ctx, pod, metav1.CreateOptions{})
	suite.Require().NoError(err, "Failed to create pod")

	// delete pod when done
	defer func() {
		err := suite.kubeClientSet.CoreV1().Pods(suite.namespace).Delete(suite.ctx, podName, metav1.DeleteOptions{})
		suite.Require().NoError(err, "Failed to delete pod")
	}()

	// start log collection
	startLogResponse, err := suite.LogCollectorServer.StartLog(suite.ctx, &log_collector.StartLogRequest{
		RunUID:      runUID,
		Selector:    "app=test",
		ProjectName: projectName,
	})
	suite.Require().NoError(err, "Failed to start log collection")
	suite.Require().True(startLogResponse.Success, "Failed to start log collection")

	// read state file
	stateFilePath := path.Join(suite.baseDir, "_metadata", "state.json")
	stateFile, err := os.Open(stateFilePath)
	suite.Require().NoError(err, "Failed to open state file")

	// read state file and make sure it is not empty
	stateFileBytes, err := io.ReadAll(stateFile)
	suite.Require().NoError(err, "Failed to read state file")
	suite.Require().NotEmpty(stateFileBytes, "State file is empty")

	// wait for logs to be collected
	suite.logger.InfoWith("Waiting for logs to be collected")
	time.Sleep(10 * time.Second)

	// mock the get logs server stream
	nopStream := &nop.GetLogsResponseStreamNop{}

	var logs []string
	startedGettingLogsTime := time.Now()

	for {

		// clear logs from mock stream
		nopStream.Logs = []byte{}

		// get logs until everything is read
		err := suite.LogCollectorServer.GetLogs(&log_collector.GetLogsRequest{
			RunUID:      runUID,
			Offset:      0,
			Size:        -1,
			ProjectName: projectName,
		}, nopStream)
		suite.Require().NoError(err, "Failed to get logs")

		// make sure logs have at least 100 lines
		logs = strings.Split(string(nopStream.Logs), "\n")
		if len(logs) >= expectedLogLines {
			break
		}
		if time.Since(startedGettingLogsTime) > 3*time.Minute {
			suite.Require().Fail("Timed out waiting to get all logs")
		}

		suite.logger.DebugWith("Waiting for more logs to be collected", "currentLogLines", len(logs))

		// let some more logs be collected
		time.Sleep(3 * time.Second)
	}

	suite.logger.InfoWith("Got logs", "logs", logs)
}

func (suite *LogCollectorTestSuite) TestStartLogFailureOnLabelSelector() {

	runUID := "dummy-uid"
	selector := "mlrun/uid=cde099c6724742859b8b2115eb767429,mlrun/class in (j, o, b),mlrun/project=default"

	// start log collection
	startLogResponse, err := suite.LogCollectorServer.StartLog(suite.ctx, &log_collector.StartLogRequest{
		RunUID:   runUID,
		Selector: selector,
	})

	suite.Require().False(startLogResponse.Success)
	suite.Require().Error(err)
}

func (suite *LogCollectorTestSuite) startLogCollectorServer(listenPort int) {
	err := framework.StartServer(suite.LogCollectorServer, listenPort, suite.logger)
	suite.Require().NoError(err, "Failed to start log collector server")
}

func (suite *LogCollectorTestSuite) getDummyPodSpec(podName string, lifeCycleSeconds int) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
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
						"-c",
						fmt.Sprintf("for i in $(seq 1 %d); do echo 'Test log line: ' $i; sleep 1; done", lifeCycleSeconds),
					},
				},
			},
		},
	}
}

func TestLogCollectorTestSuite(t *testing.T) {
	if testing.Short() {
		return
	}
	suite.Run(t, new(LogCollectorTestSuite))
}
