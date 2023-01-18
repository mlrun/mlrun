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

package test

import (
	"context"
	"os"
	"path"
	"strings"
	"testing"
	"time"

	"github.com/mlrun/mlrun/pkg/common"
	"github.com/mlrun/mlrun/pkg/framework"
	"github.com/mlrun/mlrun/pkg/services/logcollector"
	"github.com/mlrun/mlrun/proto/build/log_collector"

	"github.com/nuclio/logger"
	"github.com/nuclio/loggerus"
	"github.com/stretchr/testify/suite"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

type LogCollectorTestSuite struct {
	suite.Suite
	LogCollectorServer *logcollector.LogCollectorServer
	logger             logger.Logger
	ctx                context.Context
	kubeClientSet      kubernetes.Interface
	namespace          string
	baseDir            string
}

func (suite *LogCollectorTestSuite) SetupSuite() {
	var err error
	suite.logger, err = loggerus.NewLoggerusForTests("logcollector-test")
	suite.Require().NoError(err, "Failed to create logger")

	// create base dir
	suite.baseDir = path.Join(os.TempDir(), "/log_collector_test")
	err = os.MkdirAll(suite.baseDir, 0777)
	suite.Require().NoError(err, "Failed to create base dir")

	suite.ctx = context.Background()
	suite.namespace = "mlrun"
	stateFileUpdateIntervalStr := "5s"
	readLogWaitTime := "3s"
	monitoringInterval := "30s"
	bufferPoolSize := 30
	bufferSizeBytes := 1024
	listenPort := 8282

	// get kube config file path
	homeDir, err := os.UserHomeDir()
	suite.Require().NoError(err, "Failed to get home dir")
	kubeConfigFilePath := path.Join(homeDir, ".kube", "config")

	restConfig, err := common.GetKubernetesClientConfig(kubeConfigFilePath)
	suite.Require().NoError(err)

	suite.kubeClientSet, err = kubernetes.NewForConfig(restConfig)
	suite.Require().NoError(err)

	suite.LogCollectorServer, err = logcollector.NewLogCollectorServer(suite.logger,
		suite.namespace,
		suite.baseDir,
		kubeConfigFilePath,
		stateFileUpdateIntervalStr,
		readLogWaitTime,
		monitoringInterval,
		bufferPoolSize,
		bufferPoolSize,
		bufferSizeBytes)
	suite.Require().NoError(err, "Failed to create log collector server")

	// start log collector server in a goroutine, so it won't block the test
	go suite.startLogCollectorServer(listenPort)

	suite.logger.InfoWith("Setup completed")
}

func (suite *LogCollectorTestSuite) TearDownSuite() {

	// delete base dir and created files
	err := os.RemoveAll(suite.baseDir)
	suite.Require().NoError(err, "Failed to delete base dir")

	suite.logger.InfoWith("Tear down complete")
}

func (suite *LogCollectorTestSuite) TestLogCollector() {

	// create pod that prints logs
	podName := "test-pod"
	runUID := "some-uid"
	pod := v1.Pod{
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
						"for i in $(seq 1 100); do echo 'Test log line: ' $i; sleep 1; done",
					},
				},
			},
		},
	}

	_, err := suite.kubeClientSet.CoreV1().Pods(suite.namespace).Create(suite.ctx, &pod, metav1.CreateOptions{})
	suite.Require().NoError(err, "Failed to create pod")

	// delete pod when done
	defer func() {
		err := suite.kubeClientSet.CoreV1().Pods(suite.namespace).Delete(suite.ctx, podName, metav1.DeleteOptions{})
		suite.Require().NoError(err, "Failed to delete pod")
	}()

	// start log collection
	startLogResponse, err := suite.LogCollectorServer.StartLog(suite.ctx, &log_collector.StartLogRequest{
		RunUID:   runUID,
		Selector: "app=test",
	})

	suite.Require().NoError(err, "Failed to start log collection")
	suite.Require().True(startLogResponse.Success, "Failed to start log collection")

	// wait for logs to be collected
	suite.logger.InfoWith("Waiting for logs to be collected")
	time.Sleep(10 * time.Second)

	var logs []string
	startedGettingLogsTime := time.Now()
	offset := 0

	for {

		// get logs until everything is read
		getLogsResponse, err := suite.LogCollectorServer.GetLogs(suite.ctx, &log_collector.GetLogsRequest{
			RunUID: runUID,
			Offset: uint64(offset),
			Size:   0,
		})
		suite.Require().NoError(err, "Failed to get logs")
		suite.Require().True(getLogsResponse.Success, "Failed to get logs")

		// make sure logs have at least 100 lines
		logLines := strings.Split(string(getLogsResponse.Logs), "\n")
		suite.logger.InfoWith("Got logs", "numLines", len(logLines))
		logs = append(logs, logLines...)
		if len(logs) >= 100 {
			break
		}
		if time.Since(startedGettingLogsTime) > 2*time.Minute {
			suite.Require().Fail("Timed out waiting to get all logs")
		}
		offset += len(getLogsResponse.Logs)

		// let some more logs be collected
		time.Sleep(3 * time.Second)
	}

	suite.logger.InfoWith("Got logs", "logs", logs)
}

func (suite *LogCollectorTestSuite) startLogCollectorServer(listenPort int) {
	err := framework.StartServer(suite.LogCollectorServer, listenPort, suite.logger)
	suite.Require().NoError(err, "Failed to start log collector server")
}

func TestLogCollectorTestSuite(t *testing.T) {
	suite.Run(t, new(LogCollectorTestSuite))
}
