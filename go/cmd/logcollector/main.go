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
	"os"

	"github.com/mlrun/mlrun/pkg/common"
	"github.com/mlrun/mlrun/pkg/framework"
	"github.com/mlrun/mlrun/pkg/services/logcollector"

	"github.com/nuclio/errors"
)

func StartServer() error {

	// env vars parsing
	listenPort := flag.Int("listen-port", common.GetEnvOrDefaultInt("MLRUN_LOG_COLLECTOR_LISTEN_PORT", 8080), "GRPC listen port")
	logLevel := flag.String("log-level", common.GetEnvOrDefaultString("MLRUN_LOG_COLLECTOR_LOG_LEVEL", "debug"), "Log level (debug, info, warn, error, fatal, panic)")
	logFormatter := flag.String("log-formatter", common.GetEnvOrDefaultString("MLRUN_LOG_COLLECTOR_LOG_FORMATTER", "text"), "Log formatter (text, json)")
	baseDir := flag.String("base-dir", common.GetEnvOrDefaultString("MLRUN_LOG_COLLECTOR_BASE_DIR", "/var/mlrun/log-collector/pod-logs"), "The directory to store the logs in")
	kubeconfigPath := flag.String("kubeconfig-path", common.GetEnvOrDefaultString("MLRUN_LOG_COLLECTOR_KUBECONFIG_PATH", ""), "Path to kubeconfig file")
	stateFileUpdateInterval := flag.String("state-file-update-interval", common.GetEnvOrDefaultString("MLRUN_LOG_COLLECTOR_STATE_FILE_UPDATE_INTERVAL", "10s"), "Periodic interval for updating the state file")
	readLogWaitTime := flag.String("read-log-wait-time", common.GetEnvOrDefaultString("MLRUN_LOG_COLLECTOR_READ_LOG_WAIT_TIME", "3s"), "Wait time until trying to get more logs from the pod")
	monitoringInterval := flag.String("monitoring-interval", common.GetEnvOrDefaultString("MLRUN_LOG_COLLECTOR_MONITORING_INTERVAL", "30s"), "Periodic interval for monitoring the goroutines collecting logs")
	logCollectionbufferPoolSize := flag.Int("log-collection-buffer-pool-size", common.GetEnvOrDefaultInt("MLRUN_LOG_COLLECTOR_LOG_COLLECTION_BUFFER_POOL_SIZE", 512), "Number of buffers in the buffer pool for collecting logs")
	getLogsBufferPoolSize := flag.Int("get-logs-buffer-pool-size", common.GetEnvOrDefaultInt("MLRUN_LOG_COLLECTOR_GET_LOGS_BUFFER_POOL_SIZE", 512), "Number of buffers in the buffer pool for getting logs")
	bufferSizeBytes := flag.Int("buffer-size-bytes", common.GetEnvOrDefaultInt("MLRUN_LOG_COLLECTOR_BUFFER_SIZE_BYTES", 10485760), "Size of buffer in bytes for reading pod logs")

	// if namespace is not passed, it will be taken from env
	namespace := flag.String("namespace", "", "The namespace to collect logs from")

	flag.Parse()

	*namespace = getNamespace(*namespace)

	logger, err := framework.NewLogger("log-collector", *logLevel, *logFormatter)
	if err != nil {
		return errors.Wrap(err, "Failed to create logger")
	}
	server, err := logcollector.NewLogCollectorServer(logger,
		*namespace,
		*baseDir,
		*kubeconfigPath,
		*stateFileUpdateInterval,
		*readLogWaitTime,
		*monitoringInterval,
		*logCollectionbufferPoolSize,
		*getLogsBufferPoolSize,
		*bufferSizeBytes)
	if err != nil {
		return errors.Wrap(err, "Failed to create log collector server")
	}
	return framework.StartServer(server, *listenPort, server.Logger)
}

func getNamespace(namespaceArgument string) string {

	// if the namespace was passed in the arguments, use that
	if namespaceArgument != "" {
		return namespaceArgument
	}

	// if the namespace exists in env, use that
	if namespaceEnv := os.Getenv("MLRUN_LOG_COLLECTOR_NAMESPACE"); namespaceEnv != "" {
		return namespaceEnv
	}

	// if nothing was passed, assume "this" namespace
	return "@mlrun.selfNamespace"
}

func main() {
	err := StartServer()
	if err != nil {
		stackTrace := errors.GetErrorStackString(err, 10)
		fmt.Printf("Failed to start log collector server: %s", stackTrace)
		panic(err)
	}
}
