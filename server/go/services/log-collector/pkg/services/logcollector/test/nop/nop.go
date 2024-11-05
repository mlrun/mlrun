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

package nop

import (
	"context"

	"github.com/mlrun/mlrun-go/pkg/proto/build/log_collector"

	"google.golang.org/grpc"
)

// GetLogsResponseStreamNop is a nop implementation of the protologcollector.LogCollector_GetLogsServer interface
type GetLogsResponseStreamNop struct {
	grpc.ServerStream
	Logs []byte
}

func (m *GetLogsResponseStreamNop) Send(response *log_collector.GetLogsResponse) error {
	m.Logs = append(m.Logs, response.Logs...)
	return nil
}

func (m *GetLogsResponseStreamNop) Context() context.Context {
	return context.Background()
}

// ListRunsResponseStreamNop is a nop implementation of the protologcollector.LogCollector_ListRunsServer interface
type ListRunsResponseStreamNop struct {
	grpc.ServerStream
	RunUIDs []string
}

func (m *ListRunsResponseStreamNop) Send(response *log_collector.ListRunsResponse) error {
	m.RunUIDs = append(m.RunUIDs, response.RunUIDs...)
	return nil
}

func (m *ListRunsResponseStreamNop) Context() context.Context {
	return context.Background()
}
