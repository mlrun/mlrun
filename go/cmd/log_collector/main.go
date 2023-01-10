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

	"github.com/mlrun/mlrun/pkg/framework"
	"github.com/mlrun/mlrun/pkg/services/log_collector"

	"github.com/nuclio/errors"
)

func StartServer() error {
	listenPort := flag.Int("listen-port", 8080, "GRPC listen port")
	logLevel := flag.String("log-level", "info", "Log level (debug, info, warn, error, fatal, panic)")
	logFormatter := flag.String("log-formatter", "text", "Log formatter (text, json)")

	flag.Parse()

	logger, err := framework.NewLogger("log-collector", *logLevel, *logFormatter)
	if err != nil {
		return errors.Wrap(err, "Failed to create logger")
	}
	server, err := log_collector.NewLogCollectorServer(logger)
	if err != nil {
		return errors.Wrap(err, "Failed to create log collector server")
	}
	return framework.StartServer(server, *listenPort)
}

func main() {
	err := StartServer()
	if err != nil {
		panic(err)
	}
}
