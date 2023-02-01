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

package main

import (
	"flag"
	"fmt"

	"github.com/mlrun/mlrun/pkg/common"
	"github.com/mlrun/mlrun/pkg/common/k8s"
	"github.com/mlrun/mlrun/pkg/framework"
	"github.com/mlrun/mlrun/pkg/services/logcollector"

	"github.com/nuclio/errors"
	"k8s.io/client-go/kubernetes"
)

func StartServer() error {

	// env vars parsing
	listenPort := flag.Int("listen-port", common.GetEnvOrDefaultInt("MLRUN_LOG_COLLECTOR__LISTEN_PORT", 8080), "GRPC listen port")
	logLevel := flag.String("log-level", common.GetEnvOrDefaultString("MLRUN_LOG_COLLECTOR__LOG_LEVEL", "debug"), "Log level (debug, info, warn, error, fatal, panic)")
	logFormatter := flag.String("log-formatter", common.GetEnvOrDefaultString("MLRUN_LOG_COLLECTOR__LOG_FORMATTER", "text"), "Log formatter (text, json)")
	baseDir := flag.String("base-dir", common.GetEnvOrDefaultString("MLRUN_LOG_COLLECTOR__BASE_DIR", "/var/mlrun/logs"), "The directory to store the logs in")
	kubeconfigPath := flag.String("kubeconfig-path", common.GetEnvOrDefaultString("MLRUN_LOG_COLLECTOR__KUBECONFIG_PATH", ""), "Path to kubeconfig file")
	stateFileUpdateInterval := flag.String("state-file-update-interval", common.GetEnvOrDefaultString("MLRUN_LOG_COLLECTOR__STATE_FILE_UPDATE_INTERVAL", "10s"), "Periodic interval for updating the state file (default 10s)")
	readLogWaitTime := flag.String("read-log-wait-time", common.GetEnvOrDefaultString("MLRUN_LOG_COLLECTOR__READ_LOG_WAIT_TIME", "3s"), "Wait time until trying to get more logs from the pod (default 3s)")
	monitoringInterval := flag.String("monitoring-interval", common.GetEnvOrDefaultString("MLRUN_LOG_COLLECTOR__MONITORING_INTERVAL", "10s"), "Periodic interval for monitoring the goroutines collecting logs (default 10s)")
	logCollectionBufferPoolSize := flag.Int("log-collection-buffer-pool-size", common.GetEnvOrDefaultInt("MLRUN_LOG_COLLECTOR__LOG_COLLECTION_BUFFER_POOL_SIZE", 512), "Number of buffers in the buffer pool for collecting logs (default: 512 buffers)")
	getLogsBufferPoolSize := flag.Int("get-logs-buffer-pool-size", common.GetEnvOrDefaultInt("MLRUN_LOG_COLLECTOR__GET_LOGS_BUFFER_POOL_SIZE", 512), "Number of buffers in the buffer pool for getting logs (default: 512 buffers)")
	bufferSizeBytes := flag.Int("buffer-size-bytes", common.GetEnvOrDefaultInt("MLRUN_LOG_COLLECTOR__BUFFER_SIZE_BYTES", 10*1024*1024), "Size of buffers in the buffer pool, in bytes (default: 10MB)")
	clusterizationRole := flag.String("clusterization-role", common.GetEnvOrDefaultString("MLRUN_HTTPDB__CLUSTERIZATION__ROLE", "chief"), "The role of the log collector in the cluster (chief, worker)")

	// if namespace is not passed, it will be taken from env
	namespace := flag.String("namespace", "", "The namespace to collect logs from")

	flag.Parse()

	*namespace = k8s.ResolveRunningNamespace(*namespace)

	// initialize kubernetes client
	restConfig, err := common.GetKubernetesClientConfig(*kubeconfigPath)
	if err != nil {
		return errors.Wrap(err, "Failed to get client configuration")
	}
	kubeClientSet, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		return errors.Wrap(err, "Failed to create kubernetes client set")
	}

	logger, err := framework.NewLogger("log-collector", *logLevel, *logFormatter)
	if err != nil {
		return errors.Wrap(err, "Failed to create logger")
	}
	server, err := logcollector.NewLogCollectorServer(logger,
		*namespace,
		*baseDir,
		*stateFileUpdateInterval,
		*readLogWaitTime,
		*monitoringInterval,
		*clusterizationRole,
		kubeClientSet,
		*logCollectionBufferPoolSize,
		*getLogsBufferPoolSize,
		*bufferSizeBytes)
	if err != nil {
		return errors.Wrap(err, "Failed to create log collector server")
	}
	return framework.StartServer(server, *listenPort, server.Logger)
}

func main() {
	err := StartServer()
	if err != nil {
		stackTrace := errors.GetErrorStackString(err, 10)
		fmt.Printf("Failed to start log collector server: %s", stackTrace)
		panic(err)
	}
}
