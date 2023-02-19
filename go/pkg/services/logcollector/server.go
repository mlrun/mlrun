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
	"context"
	"fmt"
	"io"
	"math"
	"os"
	"path"
	"path/filepath"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"github.com/mlrun/mlrun/pkg/common"
	"github.com/mlrun/mlrun/pkg/common/bufferpool"
	mlruncontext "github.com/mlrun/mlrun/pkg/context"
	"github.com/mlrun/mlrun/pkg/framework"
	"github.com/mlrun/mlrun/pkg/services/logcollector/statestore"
	"github.com/mlrun/mlrun/pkg/services/logcollector/statestore/factory"
	protologcollector "github.com/mlrun/mlrun/proto/build/log_collector"

	"github.com/nuclio/errors"
	"github.com/nuclio/logger"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

type Server struct {
	*framework.AbstractMlrunGRPCServer
	namespace                    string
	baseDir                      string
	kubeClientSet                kubernetes.Interface
	stateStore                   statestore.StateStore
	inMemoryState                statestore.StateStore
	logCollectionBufferPool      bufferpool.Pool
	getLogsBufferPool            bufferpool.Pool
	logCollectionBufferSizeBytes int
	getLogsBufferSizeBytes       int
	readLogWaitTime              time.Duration
	monitoringInterval           time.Duration
	isChief                      bool
}

// NewLogCollectorServer creates a new log collector server
func NewLogCollectorServer(logger logger.Logger,
	namespace,
	baseDir,
	stateFileUpdateInterval,
	readLogWaitTime,
	monitoringInterval,
	clusterizationRole string,
	kubeClientSet kubernetes.Interface,
	logCollectionBufferPoolSize,
	getLogsBufferPoolSize,
	logCollectionBufferSizeBytes,
	getLogsBufferSizeBytes int) (*Server, error) {
	abstractServer, err := framework.NewAbstractMlrunGRPCServer(logger, nil)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to create abstract server")
	}

	// parse interval durations
	stateFileUpdateIntervalDuration, err := time.ParseDuration(stateFileUpdateInterval)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to state file updating interval")
	}
	readLogTimeoutDuration, err := time.ParseDuration(readLogWaitTime)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to parse read log wait time duration")
	}
	monitoringIntervalDuration, err := time.ParseDuration(monitoringInterval)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to parse monitoring interval duration")
	}

	stateStore, err := factory.CreateStateStore(
		statestore.KindFile,
		&statestore.Config{
			Logger:                  logger,
			StateFileUpdateInterval: stateFileUpdateIntervalDuration,
			BaseDir:                 baseDir,
		},
	)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to create state store")
	}

	inMemoryState, err := factory.CreateStateStore(statestore.KindInMemory, &statestore.Config{
		Logger: logger,
	})
	if err != nil {
		return nil, errors.Wrap(err, "Failed to create in-memory state")
	}

	isChief := clusterizationRole == "chief"

	// ensure base dir exists
	if isChief {
		if err := common.EnsureDirExists(baseDir, os.ModePerm); err != nil {
			return nil, errors.Wrap(err, "Failed to ensure base dir exists")
		}
	}

	// ensure log collection buffer size is not bigger than the default, because of the gRPC message size limit
	if getLogsBufferSizeBytes > common.DefaultGetLogsBufferSize {
		getLogsBufferSizeBytes = common.DefaultGetLogsBufferSize
	}

	// create a byte buffer pool - a pool of size `bufferPoolSize`, where each buffer is of size `bufferSizeBytes`
	logCollectionBufferPool := bufferpool.NewSizedBytePool(logCollectionBufferPoolSize, logCollectionBufferSizeBytes)
	getLogsBufferPool := bufferpool.NewSizedBytePool(getLogsBufferPoolSize, getLogsBufferSizeBytes)

	return &Server{
		AbstractMlrunGRPCServer:      abstractServer,
		namespace:                    namespace,
		baseDir:                      baseDir,
		stateStore:                   stateStore,
		inMemoryState:                inMemoryState,
		kubeClientSet:                kubeClientSet,
		readLogWaitTime:              readLogTimeoutDuration,
		monitoringInterval:           monitoringIntervalDuration,
		logCollectionBufferPool:      logCollectionBufferPool,
		getLogsBufferPool:            getLogsBufferPool,
		logCollectionBufferSizeBytes: logCollectionBufferSizeBytes,
		getLogsBufferSizeBytes:       getLogsBufferSizeBytes,
		isChief:                      isChief,
	}, nil
}

// OnBeforeStart is called before the server starts
func (s *Server) OnBeforeStart(ctx context.Context) error {
	s.Logger.DebugCtx(ctx, "Initializing Server")

	// initialize the state store (load state from file, start state file update loop)
	// if the server is not the chief, do not monitor anything
	if s.isChief {
		if err := s.stateStore.Initialize(ctx); err != nil {
			return errors.Wrap(err, "Failed to initialize state store")
		}

		// start logging monitor
		go s.monitorLogCollection(ctx)
	}

	return nil
}

// RegisterRoutes registers the server routes
func (s *Server) RegisterRoutes(ctx context.Context) {
	s.AbstractMlrunGRPCServer.RegisterRoutes(ctx)
	protologcollector.RegisterLogCollectorServer(s.Server, s)
}

// StartLog writes the log item info to the state file, gets the pod using the label selector,
// triggers `monitorPod` and `streamLogs` goroutines.
func (s *Server) StartLog(ctx context.Context, request *protologcollector.StartLogRequest) (*protologcollector.BaseResponse, error) {

	if !s.isChief {
		s.Logger.DebugWithCtx(ctx,
			"Server is not the chief, ignoring start log request",
			"runUID", request.RunUID,
			"projectName", request.ProjectName)
		return nil, nil
	}

	s.Logger.DebugWithCtx(ctx,
		"Received Start Log request",
		"RunUID", request.RunUID,
		"Selector", request.Selector)

	// to make start log idempotent, if log collection has already started for this run uid, return success
	if s.isLogCollectionRunning(ctx, request.RunUID, request.ProjectName) {
		s.Logger.DebugWithCtx(ctx,
			"Logs are already being collected for this run uid",
			"runUID", request.RunUID)
		return s.successfulBaseResponse(), nil
	}

	var pods *v1.PodList
	var err error

	s.Logger.DebugWithCtx(ctx, "Getting run pod using label selector", "selector", request.Selector)

	// list pods using label selector until a pod is found
	if err := common.RetryUntilSuccessful(15*time.Second, 3*time.Second, func() (bool, error) {
		pods, err = s.kubeClientSet.CoreV1().Pods(s.namespace).List(ctx, metav1.ListOptions{
			LabelSelector: request.Selector,
		})

		// if no pods were found, retry
		if err != nil || pods == nil || len(pods.Items) == 0 {
			return true, errors.Wrap(err, "Failed to list pods")
		}

		// if pods were found, stop retrying
		return false, nil
	}); err != nil {
		s.Logger.ErrorWithCtx(ctx,
			"Failed to get pod using label selector",
			"err", err.Error(),
			"runUID", request.RunUID,
			"selector", request.Selector)
		err := errors.Wrapf(err, "Failed to list pods for run id %s", request.RunUID)

		// if request is best-effort, return success so run will be marked as "requested logs" in the DB
		if request.BestEffort {
			return s.successfulBaseResponse(), nil
		}
		return &protologcollector.BaseResponse{
			Success:      false,
			ErrorCode:    common.ErrCodeNotFound,
			ErrorMessage: err.Error(),
		}, err
	}

	// found a pod. for now, we only assume each run has a single pod.
	pod := pods.Items[0]

	// write log item in progress to state store
	if err := s.stateStore.AddLogItem(ctx, request.RunUID, request.Selector, request.ProjectName); err != nil {
		err := errors.Wrapf(err, "Failed to add run id %s to state file", request.RunUID)
		return &protologcollector.BaseResponse{
			Success:      false,
			ErrorCode:    common.ErrCodeInternal,
			ErrorMessage: common.GetErrorStack(err, common.DefaultErrorStackDepth),
		}, err
	}

	// create a child context before calling goroutines, so it won't be canceled
	// TODO: use https://pkg.go.dev/golang.org/x/tools@v0.5.0/internal/xcontext
	logStreamCtx, cancelCtxFunc := mlruncontext.NewDetachedWithCancel(ctx)
	startedStreamingGoroutine := make(chan bool, 1)

	// stream logs to file
	go s.startLogStreaming(logStreamCtx, request.RunUID, pod.Name, request.ProjectName, startedStreamingGoroutine, cancelCtxFunc)

	// wait for the streaming goroutine to start
	<-startedStreamingGoroutine

	// add log item to in-memory state, so we can monitor it
	if err := s.inMemoryState.AddLogItem(ctx, request.RunUID, request.Selector, request.ProjectName); err != nil {
		err := errors.Wrapf(err, "Failed to add run id %s to in memory state", request.RunUID)
		return &protologcollector.BaseResponse{
			Success:      false,
			ErrorCode:    common.ErrCodeInternal,
			ErrorMessage: common.GetErrorStack(err, common.DefaultErrorStackDepth),
		}, err
	}

	s.Logger.DebugWithCtx(ctx, "Successfully started collecting log", "runUID", request.RunUID)

	return s.successfulBaseResponse(), nil
}

// GetLogs returns the log file contents of length size from an offset, for a given run id
// if the size is negative, the entire log file available is returned
func (s *Server) GetLogs(request *protologcollector.GetLogsRequest, responseStream protologcollector.LogCollector_GetLogsServer) error {

	ctx := responseStream.Context()

	s.Logger.DebugWithCtx(ctx,
		"Received Get Log request",
		"runUID", request.RunUID,
		"size", request.Size,
		"offset", request.Offset)

	// get log file path
	filePath, err := s.getLogFilePath(ctx, request.RunUID, request.ProjectName)
	if err != nil {
		s.Logger.ErrorWithCtx(ctx,
			"Failed to get log file path",
			"err", err.Error(),
			"runUID", request.RunUID)
		return errors.Wrapf(err, "Failed to get log file path for run id %s", request.RunUID)
	}

	if request.Size == 0 {
		if err := responseStream.Send(&protologcollector.GetLogsResponse{
			Success: true,
			Logs:    []byte{},
		}); err != nil {
			return errors.Wrapf(err, "Failed to send empty logs to stream for run id %s", request.RunUID)
		}
		return nil
	}

	// open log file and calc its size
	currentLogFileSize, err := common.GetFileSize(filePath)
	if err != nil {
		return errors.Wrapf(err, "Failed to get log file size for run id %s", request.RunUID)
	}

	// if size < 0 - we read only the logs we have for this moment in time starting from offset, so GetLogs will be finite.
	// otherwise, we read only the request size from the offset
	endSize := currentLogFileSize - request.Offset
	if request.Size > 0 && endSize > request.Size {
		endSize = request.Size
	}

	// if the offset is bigger than the current log file size, return empty response
	if endSize <= 0 {
		if err := responseStream.Send(&protologcollector.GetLogsResponse{
			Success: true,
			Logs:    []byte{},
		}); err != nil {
			return errors.Wrapf(err, "Failed to send empty logs to stream for run id %s", request.RunUID)
		}
		return nil
	}

	s.Logger.DebugWithCtx(ctx, "Reading logs from file", "runUID", request.RunUID)

	offset := request.Offset
	totalLogsSize := int64(0)

	// start reading the log file until we reach the end size
	for {
		chunkSize := s.getChunkSize(request.Size, endSize, offset)

		// read logs from file in chunks
		logs, err := s.readLogsFromFile(ctx, request.RunUID, filePath, offset, chunkSize)
		if err != nil {
			return errors.Wrapf(err, "Failed to read logs from file for run id %s", request.RunUID)
		}
		totalLogsSize += int64(len(logs))

		// send logs to stream
		if err := responseStream.Send(&protologcollector.GetLogsResponse{
			Success: true,
			Logs:    logs,
		}); err != nil {
			return errors.Wrapf(err, "Failed to send logs to stream for run id %s", request.RunUID)
		}

		// if we reached the end size, or the chunk is smaller than the chunk size
		// (we reached the end of the file), stop reading
		if totalLogsSize >= endSize || len(logs) < int(chunkSize) {
			break
		}

		// increase offset by the read size
		offset += int64(len(logs))
	}

	s.Logger.DebugWithCtx(ctx,
		"Successfully read logs from file",
		"runUID", request.RunUID,
		"offset", request.Offset)

	return nil
}

// HasLogs returns true if the log file exists for a given run id
func (s *Server) HasLogs(ctx context.Context, request *protologcollector.HasLogsRequest) (*protologcollector.HasLogsResponse, error) {

	// get log file path
	if _, err := s.getLogFilePath(ctx, request.RunUID, request.ProjectName); err != nil {
		if strings.Contains(errors.RootCause(err).Error(), "not found") {
			return &protologcollector.HasLogsResponse{
				Success: true,
				HasLogs: false,
			}, nil
		}
		return &protologcollector.HasLogsResponse{
			Success:      false,
			ErrorCode:    common.ErrCodeInternal,
			ErrorMessage: common.GetErrorStack(err, common.DefaultErrorStackDepth),
		}, err
	}

	return &protologcollector.HasLogsResponse{
		Success: true,
		HasLogs: true,
	}, nil
}

// StopLog stops streaming logs for a given run id by removing it from the persistent state.
// This will prevent the monitoring loop from starting logging again for this run id
func (s *Server) StopLog(ctx context.Context, request *protologcollector.StopLogRequest) (*protologcollector.BaseResponse, error) {
	if !s.isChief {
		s.Logger.DebugWithCtx(ctx,
			"Server is not the chief, ignoring stop log request",
			"project", request.Project,
			"numRunIDs", len(request.RunUIDs))
		return s.successfulBaseResponse(), nil
	}

	if request.Project == "" {
		message := "Project name must be provided"
		s.Logger.ErrorWithCtx(ctx, message)
		return &protologcollector.BaseResponse{
			Success:      false,
			ErrorCode:    0,
			ErrorMessage: message,
		}, errors.New(message)
	}

	if len(request.RunUIDs) == 0 {

		// remove entire project from persistent state
		if err := s.stateStore.RemoveProject(request.Project); err != nil {
			message := fmt.Sprintf("Failed to remove project %s from persistent state", request.Project)
			return &protologcollector.BaseResponse{
				Success:      false,
				ErrorCode:    0,
				ErrorMessage: message,
			}, errors.Wrap(err, message)
		}

		// remove entire project from in-memory state
		if err := s.inMemoryState.RemoveProject(request.Project); err != nil {
			message := fmt.Sprintf("Failed to remove project %s from in memory state", request.Project)
			return &protologcollector.BaseResponse{
				Success:      false,
				ErrorCode:    0,
				ErrorMessage: message,
			}, errors.Wrap(err, message)
		}

		return s.successfulBaseResponse(), nil
	}

	for _, runUID := range request.RunUIDs {

		// remove item from persistent state
		if err := s.stateStore.RemoveLogItem(runUID, request.Project); err != nil {
			message := fmt.Sprintf("Failed to remove item from persistent state for run id %s", runUID)
			return &protologcollector.BaseResponse{
				Success:      false,
				ErrorCode:    0,
				ErrorMessage: message,
			}, errors.Wrap(err, message)
		}

		// remove item from in-memory state
		if err := s.inMemoryState.RemoveLogItem(runUID, request.Project); err != nil {
			message := fmt.Sprintf("Failed to remove item from in memory state for run id %s", runUID)
			return &protologcollector.BaseResponse{
				Success:      false,
				ErrorCode:    0,
				ErrorMessage: message,
			}, errors.Wrap(err, message)
		}
	}

	return s.successfulBaseResponse(), nil
}

// startLogStreaming streams logs from a pod and writes them into a file
func (s *Server) startLogStreaming(ctx context.Context,
	runUID,
	podName,
	projectName string,
	startedStreamingGoroutine chan bool,
	cancelCtxFunc context.CancelFunc) {

	// in case of a panic, remove this goroutine from the in-memory state, so the
	// monitoring loop will start logging again for this runUID.
	defer func() {

		// cancel all other goroutines spawned from this one
		defer cancelCtxFunc()

		// remove this goroutine from in-memory state
		if err := s.inMemoryState.RemoveLogItem(runUID, projectName); err != nil {
			s.Logger.WarnWithCtx(ctx, "Failed to remove item from in memory state")
		}

		if err := recover(); err != nil {
			callStack := debug.Stack()
			s.Logger.ErrorWithCtx(ctx,
				"Panic caught while creating function",
				"err", err,
				"stack", string(callStack))
		}
	}()

	s.Logger.DebugWithCtx(ctx, "Starting log streaming", "runUID", runUID, "podName", podName)

	// signal "main" function that goroutine is up
	startedStreamingGoroutine <- true

	// create a log file to the pod
	logFilePath := s.resolvePodLogFilePath(projectName, runUID, podName)
	if err := common.EnsureFileExists(logFilePath); err != nil {
		s.Logger.ErrorWithCtx(ctx,
			"Failed to ensure log file",
			"runUID", runUID,
			"logFilePath", logFilePath)
		return
	}

	// open log file in read/write and append, to allow reading the logs while we write more logs to it
	openFlags := os.O_RDWR | os.O_APPEND
	file, err := os.OpenFile(logFilePath, openFlags, 0644)
	if err != nil {
		s.Logger.ErrorWithCtx(ctx, "Failed to open file", "err", err.Error(), "logFilePath", logFilePath)
		return
	}
	defer file.Close() // nolint: errcheck

	// initialize stream and error for the while loop
	var (
		stream      io.ReadCloser
		streamErr   error
		keepLogging = true
	)

	// get logs from pod, and keep the stream open (follow)
	podLogOptions := &v1.PodLogOptions{
		Follow: true,
	}
	restClientRequest := s.kubeClientSet.CoreV1().Pods(s.namespace).GetLogs(podName, podLogOptions)

	// get the log stream - if the retry times out, the monitoring loop will restart log collection for this runUID
	if err := common.RetryUntilSuccessful(1*time.Minute, 5*time.Second, func() (bool, error) {
		stream, streamErr = restClientRequest.Stream(ctx)
		if streamErr != nil {

			// if the pod is pending, retry
			if s.isPodPendingError(streamErr) {
				return true, streamErr
			}

			// an error occurred, stop retrying
			return false, streamErr
		}

		// success
		return false, nil
	}); err != nil {
		s.Logger.ErrorWithCtx(ctx,
			"Failed to get pod log stream",
			"runUID", runUID,
			"err", common.GetErrorStack(err, common.DefaultErrorStackDepth))
		return
	}
	defer stream.Close() // nolint: errcheck

	for keepLogging {

		keepLogging, err = s.streamPodLogs(ctx, runUID, file, stream)
		if err != nil {
			s.Logger.WarnWithCtx(ctx,
				"An error occurred while streaming pod logs",
				"err", common.GetErrorStack(err, common.DefaultErrorStackDepth))
		}
	}

	s.Logger.DebugWithCtx(ctx,
		"Removing item from state file",
		"runUID", runUID,
		"podName", podName)

	// remove run from state file
	if err := s.stateStore.RemoveLogItem(runUID, projectName); err != nil {
		s.Logger.WarnWithCtx(ctx, "Failed to remove log item from state file")
	}

	s.Logger.DebugWithCtx(ctx, "Finished log streaming", "runUID", runUID, "podName", podName)
}

// streamPodLogs streams logs from a pod to a file
func (s *Server) streamPodLogs(ctx context.Context,
	runUID string,
	logFile *os.File,
	stream io.ReadCloser) (bool, error) {

	// get a buffer from the pool - so we can share buffers across goroutines
	buf := s.logCollectionBufferPool.Get()
	defer s.logCollectionBufferPool.Put(buf)

	// read from the stream into the buffer
	// this is non-blocking, it will return immediately if there is nothing to read
	numBytesRead, err := stream.Read(buf)

	if numBytesRead > 0 {

		// write to file
		if _, err := logFile.Write(buf[:numBytesRead]); err != nil {
			s.Logger.WarnWithCtx(ctx, "Failed to write pod log to file",
				"err", err.Error(),
				"runUID", runUID)
			return true, errors.Wrap(err, "Failed to write pod log to file")
		}
	}

	// if error is EOF, the pod is done streaming logs (deleted/completed/failed)
	if err == io.EOF {
		s.Logger.DebugWithCtx(ctx, "Pod logs are done streaming", "runUID", runUID)
		return false, nil
	}

	// log error if occurred
	if err != nil {
		s.Logger.WarnWithCtx(ctx, "Failed to read pod log",
			"err", err.Error(),
			"runUID", runUID)

		// if error is not nil, and we didn't read anything - a real error occurred, so we stop logging
		if numBytesRead != 0 {
			return false, errors.Wrap(err, "Failed to read pod logs")
		}
	}

	// nothing happened, continue
	return true, nil
}

// resolvePodLogFilePath returns the path to the pod log file
func (s *Server) resolvePodLogFilePath(projectName, runUID, podName string) string {
	return path.Join(s.baseDir, projectName, fmt.Sprintf("%s_%s", runUID, podName))
}

// getLogFilePath returns the path to the run's latest log file
func (s *Server) getLogFilePath(ctx context.Context, runUID, projectName string) (string, error) {

	s.Logger.DebugWithCtx(ctx, "Getting log file path", "runUID", runUID)

	logFilePath := ""
	var latestModTime time.Time

	if err := common.RetryUntilSuccessful(5*time.Second, 1*time.Second, func() (bool, error) {

		// list all files in base directory
		if err := filepath.Walk(s.baseDir, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return errors.Wrapf(err, "Failed to walk path %s", path)
			}

			// if file name starts with run id, it's a log file
			if strings.HasPrefix(info.Name(), runUID) && strings.Contains(path, projectName) {

				// if it's the first file, set it as the log file path
				// otherwise, check if it's the latest modified file
				if logFilePath == "" || info.ModTime().After(latestModTime) {
					logFilePath = path
					latestModTime = info.ModTime()
				}
			}

			return nil
		}); err != nil {
			return false, errors.Wrap(err, "Failed to list files in base directory")
		}

		if logFilePath == "" {
			return false, errors.Errorf("Log file not found for run %s", runUID)
		}

		// found log file
		return false, nil

	}); err != nil {
		return "", errors.Wrap(err, "Failed to get log file path")
	}

	return logFilePath, nil
}

// readLogsFromFile reads size bytes, starting from offset, from a log file
func (s *Server) readLogsFromFile(ctx context.Context,
	runUID,
	filePath string,
	offset,
	size int64) ([]byte, error) {

	fileSize, err := common.GetFileSize(filePath)
	if err != nil {
		return nil, errors.Wrapf(err, "Failed to get file size for run id %s", runUID)
	}

	offset, size = s.validateOffsetAndSize(offset, size, fileSize)
	if size == 0 {
		s.Logger.DebugWithCtx(ctx, "No logs to return", "run_id", runUID)
		return nil, nil
	}

	// open log file for reading
	file, err := os.OpenFile(filePath, os.O_RDONLY, 0644)
	if err != nil {
		return nil, errors.Wrapf(err, "Failed to open log file for run id %s", runUID)
	}
	defer file.Close() // nolint: errcheck

	// read size bytes from offset
	buffer := s.getLogsBufferPool.Get()
	defer s.getLogsBufferPool.Put(buffer)
	if _, err := file.ReadAt(buffer, offset); err != nil {

		// if error is EOF, return empty bytes
		if err == io.EOF {
			return buffer[:size], nil
		}
		return nil, errors.Wrapf(err, "Failed to read log file for run id %s", runUID)
	}

	return buffer[:size], nil
}

// validateOffsetAndSize validates offset and size, against the file size
func (s *Server) validateOffsetAndSize(offset, size, fileSize int64) (int64, int64) {

	// if size is negative, zero, or bigger than fileSize, read the whole file or the allowed size
	if size <= 0 || size > fileSize {
		size = int64(math.Min(float64(fileSize), float64(s.getLogsBufferSizeBytes)))
	}

	// if size is bigger than what's left to read, only read the rest of the file
	if size > fileSize-offset {
		size = fileSize - offset
	}

	// if offset is bigger than file size, don't read anything.
	if offset > fileSize {
		size = 0
	}

	return offset, size
}

// monitorLogCollection makes sure log collection is done for the runs listed in the state file
func (s *Server) monitorLogCollection(ctx context.Context) {

	s.Logger.DebugWithCtx(ctx,
		"Monitoring log streaming goroutines periodically",
		"namespace", s.namespace,
		"monitoringInterval", s.monitoringInterval)

	monitoringTicker := time.NewTicker(s.monitoringInterval)

	// count the errors so we won't spam the logs
	errCount := 0

	// Check the items in the inMemoryState against the items in the state store.
	// If an item is written in the state store but not in the in memory state - call StartLog for it,
	// as the state store is the source of truth
	for range monitoringTicker.C {

		// if there are already log items in progress, call StartLog for each of them
		projectRunUIDsInProgress, err := s.stateStore.GetItemsInProgress()
		if err == nil {

			logItemsToStart := s.getLogItemsToStart(ctx, projectRunUIDsInProgress)

			for _, logItem := range logItemsToStart {
				s.Logger.DebugWithCtx(ctx, "Starting log collection for log item", "runUID", logItem.RunUID)
				if _, err := s.StartLog(ctx, &protologcollector.StartLogRequest{
					RunUID:      logItem.RunUID,
					Selector:    logItem.LabelSelector,
					ProjectName: logItem.Project,
				}); err != nil {

					// we don't fail here, as there might be other items to start log for, just log it
					s.Logger.WarnWithCtx(ctx,
						"Failed to start log collection for log item",
						"runUID", logItem.RunUID,
						"project", logItem.Project,
						"err", common.GetErrorStack(err, 10),
					)
				}
			}
		} else {

			// don't fail because we still need the server to run
			if errCount%5 == 0 {
				errCount = 0
				s.Logger.WarnWithCtx(ctx,
					"Failed to get log items in progress",
					"err", common.GetErrorStack(err, common.DefaultErrorStackDepth))
			}
			errCount++
		}
	}
}

// isLogCollectionRunning checks if log collection is running for a given runUID
func (s *Server) isLogCollectionRunning(ctx context.Context, runUID, project string) bool {
	inMemoryInProgress, err := s.inMemoryState.GetItemsInProgress()
	if err != nil {

		// this is just for sanity, as inMemoryState won't return an error
		s.Logger.WarnWithCtx(ctx,
			"Failed to get in progress items from in memory state",
			"err", err.Error())
		return false
	}

	if projectMap, exists := inMemoryInProgress.Load(project); !exists {
		return false
	} else {
		projectRunUIDsInProgress := projectMap.(*sync.Map)
		_, running := projectRunUIDsInProgress.Load(runUID)
		return running
	}
}

// getChunkSuze returns the minimum between the request size, buffer size and the remaining size to read
func (s *Server) getChunkSize(requestSize, endSize, currentOffset int64) int64 {

	chunkSize := int64(s.getLogsBufferSizeBytes)

	// if the request size is smaller than the buffer size, use the request size
	if requestSize > 0 && requestSize < chunkSize {
		chunkSize = requestSize
	}

	// if the remaining size is smaller than the buffer size, use the remaining size
	if endSize-currentOffset < chunkSize {
		chunkSize = endSize - currentOffset
	}

	return chunkSize
}

// isPodPendingError checks if the error is due to a pod pending state
func (s *Server) isPodPendingError(err error) bool {
	errString := err.Error()
	if strings.Contains(errString, "ContainerCreating") ||
		strings.Contains(errString, "PodInitializing") {
		return true
	}

	return false
}

func (s *Server) getLogItemsToStart(ctx context.Context, projectRunUIDsInProgress *sync.Map) []statestore.LogItem {
	var logItemsToStart []statestore.LogItem

	projectRunUIDsInProgress.Range(func(projectKey, runUIDsToLogItemsValue interface{}) bool {
		runUIDsToLogItems := runUIDsToLogItemsValue.(*sync.Map)

		runUIDsToLogItems.Range(func(key, value interface{}) bool {
			logItem, ok := value.(statestore.LogItem)
			if !ok {
				s.Logger.WarnWithCtx(ctx, "Failed to convert in progress item to logItem")
				return true
			}

			// check if the log streaming is already running for this runUID
			if logCollectionStarted := s.isLogCollectionRunning(ctx, logItem.RunUID, logItem.Project); !logCollectionStarted {

				// if not, add it to the list of log items to start
				logItemsToStart = append(logItemsToStart, logItem)
			}

			return true
		})
		return true
	})
	return logItemsToStart
}

func (s *Server) successfulBaseResponse() *protologcollector.BaseResponse {
	return &protologcollector.BaseResponse{
		Success: true,
	}
}
