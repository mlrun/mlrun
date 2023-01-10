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

package framework

import (
	"os"

	"github.com/nuclio/errors"
	"github.com/nuclio/loggerus"
	"github.com/sirupsen/logrus"
)

func NewLogger(name string, logLevel string, formatter string) (*loggerus.Loggerus, error) {
	var loggerInstance *loggerus.Loggerus
	var err error

	parsedLogLevel, err := logrus.ParseLevel(logLevel)
	if err != nil {
		return nil, errors.Wrapf(err, "Failed to parse log level: %s", logLevel)
	}

	switch formatter {
	case "json":
		loggerInstance, err = loggerus.NewJSONLoggerus(name, parsedLogLevel, os.Stdout)
	case "text":
		loggerInstance, err = loggerus.NewTextLoggerus(name, parsedLogLevel, os.Stdout, true, true)
	default:
		return nil, errors.Errorf("Invalid formatter: %s", formatter)
	}

	if err != nil {
		return nil, errors.Wrap(err, "Failed to create logger")
	}

	return loggerInstance, nil
}
