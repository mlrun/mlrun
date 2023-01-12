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
	"os"

	"github.com/mlrun/mlrun/pkg/common"
	"github.com/mlrun/mlrun/pkg/framework"
	"github.com/mlrun/mlrun/pkg/services/log_collector"

	"github.com/nuclio/errors"
)

func StartServer() error {

	// flag parsing
	listenPort := flag.Int("listen-port", common.GetEnvOrDefaultInt("MLRUN_LOG_COLLECTOR_LISTEN_PORT", 8080), "GRPC listen port")
	logLevel := flag.String("log-level", common.GetEnvOrDefaultString("MLRUN_LOG_COLLECTOR_LOG_LEVEL", "debug"), "Log level (debug, info, warn, error, fatal, panic)")
	logFormatter := flag.String("log-formatter", common.GetEnvOrDefaultString("MLRUN_LOG_COLLECTOR_LOG_FORMATTER", "text"), "Log formatter (text, json)")

	// if namespace is not passed, it will be taken from env
	namespace := flag.String("namespace", "", "The namespace to collect logs from")

	// env vars parsing
	baseDir := flag.String("base-dir", common.GetEnvOrDefaultString("MLRUN_LOG_COLLECTOR_BASE_DIR", "/var/mlrun/log-collector/pod-logs"), "The directory to store the logs in")
	kubeconfigPath := flag.String("kubeconfig-path", common.GetEnvOrDefaultString("MLRUN_LOG_COLLECTOR_KUBECONFIG_PATH", ""), "Path to kubeconfig file")
	stateFileUpdateInterval := flag.String("state-file-update-interval", common.GetEnvOrDefaultString("MLRUN_LOG_COLLECTOR_STATE_FILE_UPDATE_INTERVAL", "10s"), "Interval to get the logs from the pods")

	flag.Parse()

	*namespace = getNamespace(*namespace)

	logger, err := framework.NewLogger("log-collector", *logLevel, *logFormatter)
	if err != nil {
		return errors.Wrap(err, "Failed to create logger")
	}
	server, err := log_collector.NewLogCollectorServer(logger,
		*namespace,
		*baseDir,
		*kubeconfigPath,
		*stateFileUpdateInterval)
	if err != nil {
		return errors.Wrap(err, "Failed to create log collector server")
	}
	return framework.StartServer(server, *listenPort)
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
		panic(err)
	}
}
