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

package common

import "fmt"

// error codes
const (
	ErrCodeNotFound int32 = iota
	ErrCodeInternal
	ErrCodeBadRequest
)

const (
	// DefaultLogCollectionBufferSize is the default buffer size for collecting logs from pods
	DefaultLogCollectionBufferSize int = 10 * 1024 * 1024 // 10MB

	// DefaultGetLogsBufferSize is the default buffer size for reading logs
	// gRPC has a limit of 4MB, so we set it to 3.75MB in case of overhead
	DefaultGetLogsBufferSize int = 3.75 * 1024 * 1024 // 3.75MB

	// LogTimeUpdateBytesInterval is the bytes amount to read between updates of the
	// last log time in the in memory state
	LogTimeUpdateBytesInterval int = 4 * 1024 // 4KB

	// DefaultListRunsChunkSize is the default chunk size for listing runs
	DefaultListRunsChunkSize int = 10

	// DefaultErrorStackDepth is the default stack depth for errors
	DefaultErrorStackDepth = 3
)

// Custom errors

type PodStillRunningError struct {
	PodName string
}

func (e PodStillRunningError) Error() string {
	return fmt.Sprintf("Pod %s is still running", e.PodName)
}
