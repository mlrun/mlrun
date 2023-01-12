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
	"bytes"
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
	"github.com/mlrun/mlrun/proto/build/log_collector"

	"github.com/nuclio/errors"
	"github.com/nuclio/logger"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

type LogCollectorServer struct {
	*framework.AbstractMlrunGRPCServer
	namespace     string
	baseDir       string
	kubeClientSet kubernetes.Interface
	stateStore    StateStore
	logFileLocks  map[string]sync.Locker
}

func NewLogCollectorServer(logger logger.Logger,
	namespace,
	baseDir,
	kubeconfigPath,
	stateFileUpdateInterval string) (*LogCollectorServer, error) {
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
		logFileLocks:            map[string]sync.Locker{},
	}, nil
}

func (s *LogCollectorServer) OnBeforeStart(ctx context.Context) error {
	s.Logger.DebugCtx(ctx, "Initializing Server")

	// start state updating goroutine
	go s.stateStore.UpdateState(ctx)

	// if there are already log items in progress, call StartLog for each of them
	logItemsInProgress, err := s.stateStore.GetInProgress()
	if err == nil {
		for runId, logItem := range logItemsInProgress {
			s.Logger.DebugWithCtx(ctx, "Starting log collection for log item", "runId", runId)
			if _, err := s.StartLog(ctx, &log_collector.StartLogRequest{
				RunId:    runId,
				Selector: logItem.LabelSelector,
			}); err != nil {
				s.Logger.WarnWithCtx(ctx, "Failed to start log collection for log item", "runId", runId)

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
	log_collector.RegisterLogCollectorServer(s.Server, s)
}

// StartLog writes the log item info to the state file, gets the pod using the label selector,
// triggers `monitorPod` and `streamLogs` goroutines.
func (s *LogCollectorServer) StartLog(ctx context.Context, request *log_collector.StartLogRequest) (*log_collector.StartLogResponse, error) {

	s.Logger.DebugWithCtx(ctx,
		"Received Start Log request",
		"RunId", request.RunId,
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
				s.Logger.WarnWithCtx(ctx, "Failed to list pods, retrying", "runId", request.RunId)
			} else {

				// fail on error
				err := errors.Wrapf(err, "Failed to list pods for run id %s", request.RunId)
				return &log_collector.StartLogResponse{
					Success: false,
					Error:   err.Error(),
				}, err
			}
		}
	}

	// found a pod. for now, we only assume each run has a single pod.
	pod := pods.Items[0]

	// write log item in progress to file
	if err := s.stateStore.AddLogItem(ctx, request.RunId, request.Selector); err != nil {
		err := errors.Wrapf(err, "Failed to add run id %s to state file", request.RunId)
		return &log_collector.StartLogResponse{
			Success: false,
			Error:   err.Error(),
		}, err
	}

	// add lock for the run's log file
	s.logFileLocks[request.RunId] = &sync.Mutex{}

	// TODO: create a child context before calling goroutines, so it won't be canceled

	// stream logs to file
	go s.streamPodLogs(ctx, request.RunId, pod.Name)

	return &log_collector.StartLogResponse{
		Success: true,
	}, nil
}

// GetLog returns the log file contents of length size from an offset, for a given run id
func (s *LogCollectorServer) GetLog(ctx context.Context, request *log_collector.GetLogRequest) (*log_collector.GetLogResponse, error) {
	s.Logger.DebugWithCtx(ctx, "Received Get Log request", "request", request)

	// get log file path
	filePath, err := s.getLogFilePath(request.RunId)
	if err != nil {
		err := errors.Wrapf(err, "Failed to get log file path for run id %s", request.RunId)
		return &log_collector.GetLogResponse{
			Success: false,
			Error:   err.Error(),
		}, err
	}

	// open log file
	file, err := os.OpenFile(filePath, os.O_RDWR, 0644)
	if err != nil {
		err := errors.Wrapf(err, "Failed to open log file for run id %s", request.RunId)
		return &log_collector.GetLogResponse{
			Success: false,
			Error:   err.Error(),
		}, err
	}
	defer file.Close() // nolint: errcheck

	// get file size
	fileInfo, err := file.Stat()
	if err != nil {
		err := errors.Wrapf(err, "Failed to get file info for run id %s", request.RunId)
		return &log_collector.GetLogResponse{
			Success: false,
			Error:   err.Error(),
		}, err
	}

	offset, size := s.validateOffsetAndSize(request.Offset, request.Size, uint64(fileInfo.Size()))
	if size == 0 {
		s.Logger.DebugWithCtx(ctx, "No logs to return", "run_id", request.RunId)
		return &log_collector.GetLogResponse{
			Success: true,
			Log:     []byte{},
		}, nil
	}

	// read size bytes from offset
	buffer := make([]byte, size)
	if _, err := file.ReadAt(buffer, int64(offset)); err != nil {

		// if error is EOF, return empty bytes
		if err == io.EOF {
			return &log_collector.GetLogResponse{
				Success: true,
				Log:     buffer,
			}, nil
		}

		// else return error
		err := errors.Wrapf(err, "Failed to read log file for run id %s", request.RunId)
		return &log_collector.GetLogResponse{
			Success: false,
			Error:   err.Error(),
		}, err
	}

	return &log_collector.GetLogResponse{
		Success: true,
		Log:     buffer,
	}, nil
}

// streamPodLogs streams logs from a pod and writes them into a file
func (s *LogCollectorServer) streamPodLogs(ctx context.Context, runId, podName string) {

	// create a log file to the pod
	filePath := s.resolvePodLogFilePath(runId, podName)
	if err := common.EnsureFileExists(filePath); err != nil {
		s.Logger.ErrorWithCtx(ctx,
			"Failed to ensure log file",
			"runId", runId,
			"filePath", filePath)
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

	// create a buffer pool, which enables us to reuse memory without constantly allocating
	bufferPool := sync.Pool{
		New: func() interface{} {
			return new(bytes.Buffer)
		},
	}

	var buf *bytes.Buffer
	resetBuffer := func() {
		if buf != nil {
			buf.Reset()
			bufferPool.Put(buf)
		}
	}

	for {
		resetBuffer()
		buf = bufferPool.Get().(*bytes.Buffer)

		// This is blocking until there is something to read
		numBytes, err := stream.Read(buf.Bytes())

		// if error is EOF, the pod is done streaming logs (deleted/completed/failed)
		if err == io.EOF {
			s.Logger.DebugWithCtx(ctx, "Pod logs are done streaming", "runId", runId)
			resetBuffer()
			break
		}

		// nothing read, continue
		if numBytes == 0 {
			continue
		}

		// if error is not EOF, log it and continue
		if err != nil {
			s.Logger.WarnWithCtx(ctx, "Failed to read pod log",
				"error", err.Error(),
				"runId", runId)
			continue
		}

		// write log contents to file
		if err := common.WriteToFile(ctx, s.Logger, filePath, s.logFileLocks[runId], buf.Bytes()[:numBytes], true); err != nil {
			s.Logger.ErrorWithCtx(ctx,
				"Failed to write log contents to file",
				"runId", runId,
				"podName", podName)
		}
	}

	s.Logger.DebugWithCtx(ctx,
		"Removing item from state file",
		"runId", runId,
		"podName", podName)

	// remove run from state file
	if err := s.stateStore.RemoveLogItem(runId); err != nil {
		s.Logger.WarnWithCtx(ctx, "Failed to remove log item from state file")
	}

	// remove run lock
	delete(s.logFileLocks, runId)
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

func (s *LogCollectorServer) validateOffsetAndSize(offset uint64, size uint64, fileSize uint64) (uint64, uint64) {

	// set size to fileSize if size is bigger than fileSize
	if size > fileSize {
		size = fileSize
	}

	// set size to file size - offset if size is bigger than file size - offset
	if size > fileSize-offset {
		size = fileSize - offset
	}

	// if size is zero, read the whole file
	if size == 0 {
		size = fileSize
	}

	// if offset is bigger than file size, set offset to 0
	if offset > fileSize {
		offset = 0
	}

	return offset, size
}
