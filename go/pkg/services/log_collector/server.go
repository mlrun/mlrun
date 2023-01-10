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

package log_collector

import (
	"context"

	"github.com/mlrun/mlrun/pkg/framework"
	"github.com/mlrun/mlrun/proto/build/log_collector"

	"github.com/nuclio/errors"
	"github.com/nuclio/loggerus"
)

type logCollectorServer struct {
	*framework.AbstractMlrunGRPCServer
}

func (s *logCollectorServer) RegisterRoutes(ctx context.Context) {
	s.AbstractMlrunGRPCServer.RegisterRoutes(ctx)
	log_collector.RegisterLogCollectorServer(s.Server, s)
}

func (s *logCollectorServer) StartLog(ctx context.Context, request *log_collector.StartLogRequest) (*log_collector.StartLogResponse, error) {
	return &log_collector.StartLogResponse{
		Success: true,
		Error:   "",
	}, nil
}

func NewLogCollectorServer(logger *loggerus.Loggerus) (*logCollectorServer, error) {
	abstractServer, err := framework.NewAbstractMlrunGRPCServer(logger, nil)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to create abstract server")
	}
	return &logCollectorServer{
		abstractServer,
	}, nil
}
