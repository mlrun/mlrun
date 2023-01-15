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

package logcollector

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/mlrun/mlrun/pkg/common"
	"github.com/mlrun/mlrun/pkg/framework"
	protologcollector "github.com/mlrun/mlrun/proto/build/log_collector"

	"github.com/nuclio/errors"
	"github.com/nuclio/logger"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

type LogCollectorServer struct {
	*framework.AbstractMlrunGRPCServer
	namespace       string
	baseDir         string
	kubeClientSet   kubernetes.Interface
	stateStore      StateStore
	bufferPool      sync.Pool
	readLogWaitTime time.Duration
}

func NewLogCollectorServer(logger logger.Logger,
	namespace,
	baseDir,
	kubeconfigPath,
	stateFileUpdateInterval,
	readLogWaitTime string,
	bufferSizeBytes int) (*LogCollectorServer, error) {
	abstractServer, err := framework.NewAbstractMlrunGRPCServer(logger, nil)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to create abstract server")
	}

	// initialize kubernetes client
	restConfig, err := common.GetKubernetesClientConfig(kubeconfigPath)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to get client configuration")
	}
	kubeClientSet, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to create kubernetes client set")
	}

	// parse interval durations
	stateFileUpdateIntervalDuration, err := time.ParseDuration(stateFileUpdateInterval)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to parse monitoring interval")
	}

	readLogTimeoutDuration, err := time.ParseDuration(readLogWaitTime)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to parse monitoring interval")
	}

	stateStore := NewStateFile(abstractServer.Logger, path.Join(baseDir, "state.json"), stateFileUpdateIntervalDuration)

	// ensure base dir exists
	if err := common.EnsureDirExists(baseDir, os.ModeDir); err != nil {
		return nil, errors.Wrap(err, "Failed to ensure base dir exists")
	}

	return &LogCollectorServer{
		AbstractMlrunGRPCServer: abstractServer,
		namespace:               namespace,
		baseDir:                 baseDir,
		stateStore:              stateStore,
		kubeClientSet:           kubeClientSet,
		readLogWaitTime:         readLogTimeoutDuration,
		bufferPool: sync.Pool{
			New: func() interface{} {
				return make([]byte, bufferSizeBytes)
			},
		},
	}, nil
}

func (s *LogCollectorServer) OnBeforeStart(ctx context.Context) error {
	s.Logger.DebugCtx(ctx, "Initializing Server")

	// start state updating goroutine
	go s.stateStore.UpdateState(ctx)

	// if there are already log items in progress, call StartLog for each of them
	logItemsInProgress, err := s.stateStore.GetInProgress()
	if err == nil {
		for runUID, logItem := range logItemsInProgress {
			s.Logger.DebugWithCtx(ctx, "Starting log collection for log item", "runUID", runUID)
			if _, err := s.StartLog(ctx, &protologcollector.StartLogRequest{
				RunUID:   runUID,
				Selector: logItem.LabelSelector,
			}); err != nil {
				s.Logger.WarnWithCtx(ctx, "Failed to start log collection for log item", "runUID", runUID)

				// we don't fail here, as there might be other items
				continue
			}
		}
	} else {

		// don't fail because we still need the server to run
		s.Logger.WarnWithCtx(ctx, "Failed to get log items in progress")
	}

	return nil
}

func (s *LogCollectorServer) RegisterRoutes(ctx context.Context) {
	s.AbstractMlrunGRPCServer.RegisterRoutes(ctx)
	protologcollector.RegisterLogCollectorServer(s.Server, s)
}

// StartLog writes the log item info to the state file, gets the pod using the label selector,
// triggers `monitorPod` and `streamLogs` goroutines.
func (s *LogCollectorServer) StartLog(ctx context.Context, request *protologcollector.StartLogRequest) (*protologcollector.StartLogResponse, error) {

	s.Logger.DebugWithCtx(ctx,
		"Received Start Log request",
		"RunUID", request.RunUID,
		"Selector", request.Selector)

	var pods *v1.PodList
	var err error

	// list pods using label selector until a pod is found
	errCount := 0
	for pods == nil || len(pods.Items) == 0 {
		pods, err = s.kubeClientSet.CoreV1().Pods(s.namespace).List(ctx, metav1.ListOptions{
			LabelSelector: request.Selector,
		})
		if err != nil {

			// we retry 3 times, as k8s might have some issues
			if errCount <= 3 {
				errCount++
				s.Logger.WarnWithCtx(ctx, "Failed to list pods, retrying", "runId", request.RunUID)
			} else {

				// fail on error
				err := errors.Wrapf(err, "Failed to list pods for run id %s", request.RunUID)
				return &protologcollector.StartLogResponse{
					Success: false,
					Error:   err.Error(),
				}, err
			}
		}
	}

	// found a pod. for now, we only assume each run has a single pod.
	pod := pods.Items[0]

	// write log item in progress to file
	if err := s.stateStore.AddLogItem(ctx, request.RunUID, request.Selector); err != nil {
		err := errors.Wrapf(err, "Failed to add run id %s to state file", request.RunUID)
		return &protologcollector.StartLogResponse{
			Success: false,
			Error:   err.Error(),
		}, err
	}

	// TODO: create a child context before calling goroutines, so it won't be canceled

	// stream logs to file
	go s.startLogStreaming(ctx, request.RunUID, pod.Name)

	return &protologcollector.StartLogResponse{
		Success: true,
	}, nil
}

// GetLogs returns the log file contents of length size from an offset, for a given run id
func (s *LogCollectorServer) GetLogs(ctx context.Context, request *protologcollector.GetLogsRequest) (*protologcollector.GetLogsResponse, error) {
	s.Logger.DebugWithCtx(ctx, "Received Get Log request", "request", request)

	// get log file path
	filePath, err := s.getLogFilePath(request.RunUID)
	if err != nil {
		err := errors.Wrapf(err, "Failed to get log file path for run id %s", request.RunUID)
		return &protologcollector.GetLogsResponse{
			Success: false,
			Error:   err.Error(),
		}, err
	}

	// read log from file
	buffer, err := s.readLogsFromFile(ctx, request.RunUID, filePath, request.Offset, request.Size)
	if err != nil {
		err := errors.Wrapf(err, "Failed to read logs for run id %s", request.RunUID)
		return &protologcollector.GetLogsResponse{
			Success: false,
			Error:   err.Error(),
		}, err
	}

	return &protologcollector.GetLogsResponse{
		Success: true,
		Logs:    buffer,
	}, nil
}

// startLogStreaming streams logs from a pod and writes them into a file
func (s *LogCollectorServer) startLogStreaming(ctx context.Context, runId, podName string) {

	// create a log file to the pod
	logFilePath := s.resolvePodLogFilePath(runId, podName)
	if err := common.EnsureFileExists(logFilePath); err != nil {
		s.Logger.ErrorWithCtx(ctx,
			"Failed to ensure log file",
			"runId", runId,
			"logFilePath", logFilePath)
		return
	}

	// get logs from pod, and keep the stream open (follow)
	podLogOptions := &v1.PodLogOptions{
		Follow: true,
	}
	restClientRequest := s.kubeClientSet.CoreV1().Pods(s.namespace).GetLogs(podName, podLogOptions)

	// initialize stream and error for the while loop
	var (
		streamErr error
		stream    io.ReadCloser
	)

	// stream logs - retry if failed
	for {
		stream, streamErr = restClientRequest.Stream(ctx)
		if streamErr == nil {
			break
		}
		s.Logger.WarnWithCtx(ctx, "Failed to get pod log read/closer", "runId", runId)
		time.Sleep(1 * time.Second)
	}
	defer stream.Close() // nolint: errcheck

	for {
		keepLogging, err := s.streamPodLogs(ctx, runId, logFilePath, stream)
		if err != nil {
			s.Logger.WarnWithCtx(ctx, "An error occurred while streaming pod logs", "err", err)
		}
		if keepLogging {
			continue
		}
		break
	}

	s.Logger.DebugWithCtx(ctx,
		"Removing item from state file",
		"runId", runId,
		"podName", podName)

	// remove run from state file
	if err := s.stateStore.RemoveLogItem(runId); err != nil {
		s.Logger.WarnWithCtx(ctx, "Failed to remove log item from state file")
	}
}

func (s *LogCollectorServer) streamPodLogs(ctx context.Context,
	runId,
	logFilePath string,
	stream io.ReadCloser) (bool, error) {

	// create a reader from the stream
	streamReader := bufio.NewReader(stream)

	// peek into the stream, and wait until there is something to read from it
	for {
		peekBuf, err := streamReader.Peek(1)
		if err != nil {
			s.Logger.WarnWithCtx(ctx, "Failed to peek into stream", "runId", runId)
		}
		if len(peekBuf) > 0 {
			break
		}
		time.Sleep(s.readLogWaitTime)
	}

	// open log file
	openFlags := os.O_RDWR | os.O_APPEND
	file, err := os.OpenFile(logFilePath, openFlags, 0644)
	if err != nil {
		s.Logger.WarnWithCtx(ctx, "Failed to open file", "err", err, "logFilePath", logFilePath)
		return true, err
	}
	defer file.Close() // nolint: errcheck

	// get a buffer from the pool
	buf := s.bufferPool.Get().([]byte)
	defer s.bufferPool.Put(&buf)

	// copy the stream into the file using the buffer.
	// this is blocking until there is something to read / buffer is full.
	numBytesWritten, err := io.CopyBuffer(file, streamReader, buf)

	// if error is EOF, the pod is done streaming logs (deleted/completed/failed)
	if err == io.EOF {
		s.Logger.DebugWithCtx(ctx, "Pod logs are done streaming", "runId", runId)
		return false, nil
	}

	// nothing read, continue
	if numBytesWritten == 0 {
		return true, nil
	}

	// if error is not EOF, log it and continue
	if err != nil {
		s.Logger.WarnWithCtx(ctx, "Failed to read pod log",
			"error", err.Error(),
			"runId", runId)
		return true, errors.Wrap(err, "Failed to read pod logs")
	}

	// sanity
	return true, nil
}

// resolvePodLogFilePath returns the path to the pod log file
func (s *LogCollectorServer) resolvePodLogFilePath(runId, podName string) string {
	return path.Join(s.baseDir, "logs", fmt.Sprintf("%s_%s.log", runId, podName))
}

func (s *LogCollectorServer) getLogFilePath(runId string) (string, error) {

	logFilePath := ""
	var latestModTime time.Time

	// list all files in base directory
	err := filepath.Walk(s.baseDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// if file name starts with run id, it's a log file
		if strings.HasPrefix(info.Name(), runId) {

			// if it's the first file, set it as the log file path
			// otherwise, check if it's the latest modified file
			if logFilePath == "" || info.ModTime().After(latestModTime) {
				logFilePath = path
				latestModTime = info.ModTime()
			}
		}

		return nil
	})
	if err != nil {
		return "", errors.Wrap(err, "Failed to list files in base directory")
	}

	// if no log file was found, return error
	if logFilePath == "" {
		return "", errors.Errorf("Failed to find log file for run id %s", runId)
	}

	return logFilePath, nil
}

func (s *LogCollectorServer) readLogsFromFile(ctx context.Context,
	runUID,
	filePath string,
	offset uint64,
	size int64) ([]byte, error) {

	// open log file
	file, err := os.OpenFile(filePath, os.O_RDWR, 0644)
	if err != nil {
		return nil, errors.Wrapf(err, "Failed to open log file for run id %s", runUID)
	}
	defer file.Close() // nolint: errcheck

	// get file size
	fileInfo, err := file.Stat()
	if err != nil {
		return nil, errors.Wrapf(err, "Failed to get file info for run id %s", runUID)
	}

	offset, size = s.validateOffsetAndSize(offset, size, fileInfo.Size())
	if size == 0 {
		s.Logger.DebugWithCtx(ctx, "No logs to return", "run_id", runUID)
		return nil, nil
	}

	// read size bytes from offset
	buffer := make([]byte, size)
	if _, err := file.ReadAt(buffer, int64(offset)); err != nil {

		// if error is EOF, return empty bytes
		if err == io.EOF {
			return buffer, nil
		}

		// else return error
		err := errors.Wrapf(err, "Failed to read log file for run id %s", runUID)
		return nil, err
	}

	return buffer, nil
}

func (s *LogCollectorServer) validateOffsetAndSize(offset uint64, size, fileSize int64) (uint64, int64) {

	// if size is negative, zero, or bigger than fileSize, read the whole file
	if size <= 0 || size > fileSize {
		size = fileSize
	}

	// if offset is bigger than file size, set offset to 0
	if int64(offset) > fileSize {
		offset = 0
	}

	// set size to file size - offset if size is bigger than file size - offset
	if size > fileSize-int64(offset) {
		size = fileSize - int64(offset)
	}

	return offset, size
}
