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
	"syscall"
	"time"

	"github.com/mlrun/mlrun/pkg/common"
	"github.com/mlrun/mlrun/pkg/common/bufferpool"
	"github.com/mlrun/mlrun/pkg/framework"
	"github.com/mlrun/mlrun/pkg/services/logcollector/statestore"
	"github.com/mlrun/mlrun/pkg/services/logcollector/statestore/factory"
	protologcollector "github.com/mlrun/mlrun/proto/build/log_collector"

	"github.com/nuclio/errors"
	"github.com/nuclio/logger"
	"github.com/samber/lo"
	"golang.org/x/sync/errgroup"
	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

type Server struct {
	*framework.AbstractMlrunGRPCServer
	namespace        string
	baseDir          string
	kubeClientSet    kubernetes.Interface
	isChief          bool
	advancedLogLevel int

	// the state manifest determines which runs' logs should be collected, and is persisted to a file
	stateManifest statestore.StateStore
	// the current state has the actual runs that are currently being collected, and is not persisted
	currentState statestore.StateStore

	// buffer pools
	logCollectionBufferPool      bufferpool.Pool
	logCollectionBufferSizeBytes int
	getLogsBufferPool            bufferpool.Pool
	getLogsBufferSizeBytes       int
	logTimeUpdateBytesInterval   int

	// start logs finding pods timeout
	startLogsFindingPodsTimeout  time.Duration
	startLogsFindingPodsInterval time.Duration

	// interval durations
	readLogWaitTime    time.Duration
	monitoringInterval time.Duration

	listRunsChunkSize int
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
	getLogsBufferSizeBytes,
	logTimeUpdateBytesInterval,
	advancedLogLevel,
	listRunsChunkSize int) (*Server, error) {
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

	fileStateStore, err := factory.CreateStateStore(
		statestore.KindFile,
		&statestore.Config{
			Logger:                  logger,
			StateFileUpdateInterval: stateFileUpdateIntervalDuration,
			BaseDir:                 baseDir,
			AdvancedLogLevel:        advancedLogLevel,
		},
	)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to create state store")
	}

	inMemoryStateStore, err := factory.CreateStateStore(statestore.KindInMemory, &statestore.Config{
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

	if listRunsChunkSize <= 0 {
		listRunsChunkSize = common.DefaultListRunsChunkSize
	}

	return &Server{
		AbstractMlrunGRPCServer:      abstractServer,
		namespace:                    namespace,
		baseDir:                      baseDir,
		stateManifest:                fileStateStore,
		currentState:                 inMemoryStateStore,
		kubeClientSet:                kubeClientSet,
		readLogWaitTime:              readLogTimeoutDuration,
		monitoringInterval:           monitoringIntervalDuration,
		logCollectionBufferPool:      logCollectionBufferPool,
		getLogsBufferPool:            getLogsBufferPool,
		logCollectionBufferSizeBytes: logCollectionBufferSizeBytes,
		getLogsBufferSizeBytes:       getLogsBufferSizeBytes,
		logTimeUpdateBytesInterval:   logTimeUpdateBytesInterval,
		isChief:                      isChief,
		startLogsFindingPodsInterval: 3 * time.Second,
		startLogsFindingPodsTimeout:  15 * time.Second,
		advancedLogLevel:             advancedLogLevel,
		listRunsChunkSize:            listRunsChunkSize,
	}, nil
}

// OnBeforeStart is called before the server starts
func (s *Server) OnBeforeStart(ctx context.Context) error {
	s.Logger.DebugCtx(ctx, "Initializing Server")

	// if the server is not the chief, do not monitor anything
	if s.isChief {

		// initialize the state manifest (load state from file, start state file update loop)
		if err := s.stateManifest.Initialize(ctx); err != nil {
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
func (s *Server) StartLog(ctx context.Context,
	request *protologcollector.StartLogRequest) (*protologcollector.BaseResponse, error) {

	if !s.isChief {
		s.Logger.DebugWithCtx(ctx,
			"Server is not the chief, ignoring start log request",
			"runUID", request.RunUID,
			"projectName", request.ProjectName)
		return nil, nil
	}

	s.Logger.DebugWithCtx(ctx,
		"Received start log request",
		"RunUID", request.RunUID,
		"Selector", request.Selector)

	// to make start log idempotent, if log collection has already started for this run uid, return success
	if s.isLogCollectionRunning(ctx, request.RunUID, request.ProjectName) {
		s.Logger.DebugWithCtx(ctx,
			"Logs are already being collected for this run uid",
			"runUID", request.RunUID)
		return s.successfulBaseResponse(), nil
	}

	var pod v1.Pod

	s.Logger.DebugWithCtx(ctx, "Getting run pod using label selector", "selector", request.Selector)

	// list pods using label selector until a pod is found
	if err := common.RetryUntilSuccessful(
		s.startLogsFindingPodsTimeout,
		s.startLogsFindingPodsInterval,
		func() (bool, error) {
			pods, err := s.kubeClientSet.CoreV1().Pods(s.namespace).List(ctx, metav1.ListOptions{
				LabelSelector: request.Selector,
			})

			// if no pods were found, retry
			if err != nil {
				return true, errors.Wrap(err, "Failed to list pods")
			} else if pods == nil || len(pods.Items) == 0 {
				return true, errors.Errorf("No pods found for run uid '%s'", request.RunUID)
			}

			// found pods. we take the first pod because each run has a single pod.
			pod = pods.Items[0]

			// fail if pod is pending, as we cannot stream logs from a pending pod
			if pod.Status.Phase == v1.PodPending {
				return true, errors.Errorf("Pod '%s' is in pending state", pod.Name)
			}

			// all good, stop retrying
			return false, nil
		}); err != nil {

		// if request is best-effort, return success so run will be marked as "requested logs" in the DB
		if request.BestEffort {
			s.Logger.WarnWithCtx(ctx,
				"Failed to get pod using label selector, but request is best effort, returning success",
				"err", errors.RootCause(err).Error(),
				"runUID", request.RunUID,
				"projectName", request.ProjectName,
				"selector", request.Selector)
			return s.successfulBaseResponse(), nil
		}

		var lastErr = err

		// this is simply a timeout err, we want the root cause
		if errors.Is(err, common.ErrRetryUntilSuccessfulTimeout) {
			lastErr = errors.RootCause(lastErr)
		}
		s.Logger.ErrorWithCtx(ctx,
			"Failed to get pod using label selector",
			"err", common.GetErrorStack(lastErr, common.DefaultErrorStackDepth),
			"runUID", request.RunUID,
			"projectName", request.ProjectName,
			"selector", request.Selector)

		err := errors.Wrapf(lastErr, "Failed to find run '%s' pods", request.RunUID)
		return &protologcollector.BaseResponse{
			Success:      false,
			ErrorCode:    common.ErrCodeNotFound,
			ErrorMessage: err.Error(),
		}, err
	}

	// write log item in progress to state store
	if err := s.stateManifest.AddLogItem(ctx, request.RunUID, request.Selector, request.ProjectName); err != nil {
		err := errors.Wrapf(err, "Failed to add run id %s to state file", request.RunUID)
		return &protologcollector.BaseResponse{
			Success:      false,
			ErrorCode:    common.ErrCodeInternal,
			ErrorMessage: common.GetErrorStack(err, common.DefaultErrorStackDepth),
		}, err
	}

	startedStreamingGoroutine := make(chan bool, 1)

	// stream logs to file
	go s.startLogStreaming(context.WithoutCancel(ctx),
		request.RunUID,
		pod.Name,
		request.ProjectName,
		request.LastLogTime,
		startedStreamingGoroutine)

	// wait for the streaming goroutine to start
	<-startedStreamingGoroutine

	// add log item to current state, so we can monitor it
	if err := s.currentState.AddLogItem(ctx, request.RunUID, request.Selector, request.ProjectName); err != nil {
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
		"Received get log request",
		"runUID", request.RunUID,
		"size", request.Size,
		"offset", request.Offset)

	// if size is 0, return empty logs
	if request.Size == 0 {
		if err := responseStream.Send(&protologcollector.GetLogsResponse{
			Success: true,
			Logs:    []byte{},
		}); err != nil {
			return errors.Wrapf(err, "Failed to send empty logs to stream for run id %s", request.RunUID)
		}
		return nil
	}

	if request.Offset < 0 {
		return errors.New("Offset cannot be negative")
	}

	// get log file path
	filePath, err := s.getLogFilePath(ctx, request.RunUID, request.ProjectName)
	if err != nil {
		s.Logger.ErrorWithCtx(ctx,
			"Failed to get log file path",
			"err", common.GetErrorStack(err, common.DefaultErrorStackDepth),
			"runUID", request.RunUID)
		return errors.Wrapf(err, "Failed to get log file path for run id %s", request.RunUID)
	}

	// open log file and calc its size
	currentLogFileSize, err := common.GetFileSize(filePath)
	if err != nil {
		return errors.Wrapf(err, "Failed to get log file size for run id %s", request.RunUID)
	}

	// if the offset is bigger than the current log file size, return empty response
	// this happens when client is ready for next iteration while server did not collect the logs just yet
	// we send empty response to ensure client will not get stuck and keep retrying
	if currentLogFileSize-request.Offset <= 0 {
		if err := responseStream.Send(&protologcollector.GetLogsResponse{
			Success: true,
			Logs:    []byte{},
		}); err != nil {
			return errors.Wrapf(err, "Failed to send empty logs to stream for run id %s", request.RunUID)
		}
		return nil
	}

	var endIndex = currentLogFileSize

	// if request size is positive, read only the requested size, add the offset to ensure we read
	// all requested logs
	if request.Size > 0 {
		endIndex = request.Size + request.Offset
	}

	// if the end size is bigger than the current log file size, read until the end of the file
	if endIndex > currentLogFileSize {
		endIndex = currentLogFileSize
	}

	s.Logger.DebugWithCtx(ctx, "Reading logs from file", "runUID", request.RunUID)

	offset := request.Offset
	totalLogsSize := int64(0)

	// start reading the log file until we reach the end size
	for {
		chunkSize := s.getChunkSize(request.Offset, totalLogsSize, endIndex)

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
		if totalLogsSize+request.Offset >= endIndex || len(logs) < int(chunkSize) {
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

// GetLogSize returns the size of the log file for a given run id
func (s *Server) GetLogSize(ctx context.Context, request *protologcollector.GetLogSizeRequest) (*protologcollector.GetLogSizeResponse, error) {
	s.Logger.DebugWithCtx(ctx,
		"Received get log size request",
		"runUID", request.RunUID,
		"project", request.ProjectName)

	// get log file path
	filePath, err := s.getLogFilePath(ctx, request.RunUID, request.ProjectName)
	if err != nil {
		if strings.Contains(errors.RootCause(err).Error(), "not found") {

			// if the log file is not found, return false but no error
			s.Logger.DebugWithCtx(ctx,
				"Log file not found",
				"runUID", request.RunUID,
				"projectName", request.ProjectName)
			return &protologcollector.GetLogSizeResponse{
				Success: true,
				LogSize: -1,
			}, nil
		}

		// if there was an error, return it
		s.Logger.ErrorWithCtx(ctx,
			"Failed to check if log file exists",
			"err", common.GetErrorStack(err, common.DefaultErrorStackDepth),
			"runUID", request.RunUID,
			"projectName", request.ProjectName)

		// do not return the 'err' itself, so that mlrun api would catch the response
		// and will resolve the response on its own.
		return &protologcollector.GetLogSizeResponse{
			Success:      false,
			ErrorCode:    common.ErrCodeInternal,
			ErrorMessage: common.GetErrorStack(err, common.DefaultErrorStackDepth),
		}, nil
	}

	// open log file and calc its size
	currentLogFileSize, err := common.GetFileSize(filePath)
	if err != nil {
		s.Logger.ErrorWithCtx(ctx,
			"Failed to get log file size",
			"err", common.GetErrorStack(err, common.DefaultErrorStackDepth),
			"runUID", request.RunUID,
			"projectName", request.ProjectName)
		err = errors.Wrapf(err, "Failed to get log file size for run id %s", request.RunUID)
		return &protologcollector.GetLogSizeResponse{
			Success:      false,
			ErrorCode:    common.ErrCodeInternal,
			ErrorMessage: common.GetErrorStack(err, common.DefaultErrorStackDepth),
		}, nil
	}

	return &protologcollector.GetLogSizeResponse{
		Success: true,
		LogSize: currentLogFileSize,
	}, nil
}

// StopLogs stops streaming logs for a given run id by removing it from the persistent state.
// This will prevent the monitoring loop from starting logging again for this run id
func (s *Server) StopLogs(ctx context.Context, request *protologcollector.StopLogsRequest) (*protologcollector.BaseResponse, error) {
	if !s.isChief {
		s.Logger.DebugWithCtx(ctx,
			"Server is not the chief, ignoring stop log request",
			"project", request.Project,
			"numRunIDs", len(request.RunUIDs))
		return s.successfulBaseResponse(), nil
	}

	// validate project name
	if request.Project == "" {
		message := "Project name must be provided"
		s.Logger.ErrorWithCtx(ctx, message)
		return &protologcollector.BaseResponse{
			Success:      false,
			ErrorCode:    common.ErrCodeBadRequest,
			ErrorMessage: message,
		}, errors.New(message)
	}

	// if no run uids were provided, remove the entire project from the state
	if len(request.RunUIDs) == 0 {

		// remove entire project from state manifest
		if err := s.stateManifest.RemoveProject(request.Project); err != nil {
			message := fmt.Sprintf("Failed to remove project %s from state manifest", request.Project)
			return &protologcollector.BaseResponse{
				Success:      false,
				ErrorCode:    common.ErrCodeInternal,
				ErrorMessage: message,
			}, errors.Wrap(err, message)
		}

		// remove entire project from current state
		if err := s.currentState.RemoveProject(request.Project); err != nil {
			message := fmt.Sprintf("Failed to remove project %s from in memory state", request.Project)
			return &protologcollector.BaseResponse{
				Success:      false,
				ErrorCode:    common.ErrCodeInternal,
				ErrorMessage: message,
			}, errors.Wrap(err, message)
		}

		return s.successfulBaseResponse(), nil
	}

	s.Logger.DebugWithCtx(ctx,
		"Stopping logs",
		"project", request.Project,
		"numRunIDs", len(request.RunUIDs))

	// remove each run uid from the state
	for _, runUID := range request.RunUIDs {

		// remove item from state manifest
		if err := s.stateManifest.RemoveLogItem(ctx, runUID, request.Project); err != nil {
			message := fmt.Sprintf("Failed to remove item from state manifest for run id %s", runUID)
			return &protologcollector.BaseResponse{
				Success:      false,
				ErrorCode:    common.ErrCodeInternal,
				ErrorMessage: message,
			}, errors.Wrap(err, message)
		}

		// remove item from current state
		if err := s.currentState.RemoveLogItem(ctx, runUID, request.Project); err != nil {
			message := fmt.Sprintf("Failed to remove item from in memory state for run id %s", runUID)
			return &protologcollector.BaseResponse{
				Success:      false,
				ErrorCode:    common.ErrCodeInternal,
				ErrorMessage: message,
			}, errors.Wrap(err, message)
		}
	}

	return s.successfulBaseResponse(), nil
}

// DeleteLogs deletes the log file for a given run id or project
func (s *Server) DeleteLogs(ctx context.Context, request *protologcollector.StopLogsRequest) (*protologcollector.BaseResponse, error) {

	// validate project name
	if request.Project == "" {
		message := "Project name must be provided"
		s.Logger.ErrorWithCtx(ctx, message)
		return &protologcollector.BaseResponse{
			Success:      false,
			ErrorCode:    common.ErrCodeBadRequest,
			ErrorMessage: message,
		}, errors.New(message)
	}

	// if no run uids were provided, delete the entire project's logs
	if len(request.RunUIDs) == 0 {

		s.Logger.DebugWithCtx(ctx,
			"Deleting all project logs",
			"project", request.Project)

		// remove entire project from persistent state
		if err := s.deleteProjectLogs(request.Project); err != nil {
			message := fmt.Sprintf("Failed to delete project logs for project %s", request.Project)
			return &protologcollector.BaseResponse{
				Success:      false,
				ErrorCode:    common.ErrCodeInternal,
				ErrorMessage: message,
			}, errors.Wrap(err, message)
		}

		s.Logger.DebugWithCtx(ctx,
			"Successfully deleted all project logs",
			"project", request.Project)

		return s.successfulBaseResponse(), nil
	}

	s.Logger.DebugWithCtx(ctx,
		"Deleting logs",
		"project", request.Project,
		"numRunIDs", len(request.RunUIDs))

	errGroup, _ := errgroup.WithContext(ctx)
	errGroup.SetLimit(10)
	var failedToDeleteRunUIDs []string
	for _, runUID := range request.RunUIDs {
		runUID := runUID
		errGroup.Go(func() error {

			// delete the run's log file
			if err := s.deleteRunLogFiles(ctx, runUID, request.Project); err != nil {
				failedToDeleteRunUIDs = append(failedToDeleteRunUIDs, runUID)
				return errors.Wrapf(err, "Failed to delete log files for run %s", runUID)
			}
			return nil
		})
	}

	if err := errGroup.Wait(); err != nil {
		message := fmt.Sprintf("Failed to remove logs for runs: %v", failedToDeleteRunUIDs)
		return &protologcollector.BaseResponse{
			Success:      false,
			ErrorCode:    common.ErrCodeInternal,
			ErrorMessage: message,
		}, errors.Wrap(err, message)
	}

	return s.successfulBaseResponse(), nil
}

// ListRunsInProgress returns a list of runs that are currently being collected
func (s *Server) ListRunsInProgress(request *protologcollector.ListRunsRequest, responseStream protologcollector.LogCollector_ListRunsInProgressServer) error {
	ctx := responseStream.Context()

	s.Logger.DebugWithCtx(ctx,
		"Received list runs in progress request",
		"project", request.Project)

	// get all runs in progress from the state manifest
	logItemsInProgress, err := s.stateManifest.GetItemsInProgress()
	if err != nil {
		message := "Failed to list runs in progress from state manifest"
		s.Logger.ErrorWithCtx(ctx, message)
		return errors.Wrap(err, message)
	}

	runsInProgress, err := s.getRunUIDsInProgress(ctx, logItemsInProgress, request.Project)
	if err != nil {
		message := "Failed to list runs in progress"
		s.Logger.ErrorWithCtx(ctx, message)
		return errors.Wrap(err, message)

	}

	// get all runs in progress from the current state, and add merge them with the runs from the state manifest
	// this can only happen if some voodoo occurred after the server restarted
	logItemsInProgressCurrentState, err := s.currentState.GetItemsInProgress()
	if err != nil {
		message := "Failed to get ms in progress from current state"
		s.Logger.ErrorWithCtx(ctx, message)
		return errors.Wrap(err, message)
	}

	runsInProgressCurrentState, err := s.getRunUIDsInProgress(ctx, logItemsInProgressCurrentState, request.Project)
	if err != nil {
		message := "Failed to list runs in progress from current state"
		s.Logger.ErrorWithCtx(ctx, message)
		return errors.Wrap(err, message)

	}

	// merge the two maps
	for _, runUID := range runsInProgressCurrentState {
		if !lo.Contains[string](runsInProgress, runUID) {
			runsInProgress = append(runsInProgress, runUID)
		}
	}

	// send empty response if no runs are in progress
	if len(runsInProgress) == 0 {
		s.Logger.DebugWithCtx(ctx, "No runs in progress to list")
		if err := responseStream.Send(&protologcollector.ListRunsResponse{
			RunUIDs: []string{},
		}); err != nil {
			return errors.Wrapf(err, "Failed to send empty response to stream")
		}
		return nil
	}

	// send each run in progress to the stream in chunks of 10 due to gRPC message size limit
	for i := 0; i < len(runsInProgress); i += s.listRunsChunkSize {
		endIndex := i + s.listRunsChunkSize
		if endIndex > len(runsInProgress) {
			endIndex = len(runsInProgress)
		}

		if err := responseStream.Send(&protologcollector.ListRunsResponse{
			RunUIDs: runsInProgress[i:endIndex],
		}); err != nil {
			return errors.Wrapf(err, "Failed to send runs in progress to stream")
		}
	}

	return nil
}

// startLogStreaming streams logs from a pod and writes them into a file
func (s *Server) startLogStreaming(ctx context.Context,
	runUID,
	podName,
	projectName string,
	lastLogTime int64,
	startedStreamingGoroutine chan bool) {

	// in case of a panic, remove this goroutine from the current state, so the
	// monitoring loop will start logging again for this runUID.
	defer func() {

		// update last log time in state manifest, so we can start from the last log time
		if err := s.stateManifest.UpdateLastLogTime(runUID, projectName, lastLogTime); err != nil {
			// if the pod ended properly, the run will be removed from the state file and the update will fail
			// we can ignore this error and continue
			s.Logger.WarnWithCtx(ctx,
				"Failed to update last log time, run may have ended",
				"runUID", runUID,
				"err", err.Error())
		}

		// remove this goroutine from in-current state
		if err := s.currentState.RemoveLogItem(ctx, runUID, projectName); err != nil {
			s.Logger.WarnWithCtx(ctx,
				"Failed to remove item from in memory state",
				"runUID", runUID,
				"err", err.Error())
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
	logFilePath := s.resolveRunLogFilePath(projectName, runUID)
	if err := common.EnsureFileExists(logFilePath); err != nil {
		s.Logger.ErrorWithCtx(ctx,
			"Failed to ensure log file",
			"runUID", runUID,
			"logFilePath", logFilePath)
		return
	}

	// open log file in read/write and append, to allow reading the logs while we write more logs to it
	openFlags := os.O_RDWR | os.O_APPEND
	file, err := os.OpenFile(logFilePath, openFlags, 0600)
	if err != nil {
		s.Logger.ErrorWithCtx(ctx,
			"Failed to open file",
			"err", err.Error(),
			"logFilePath", logFilePath)
		return
	}
	defer file.Close() // nolint: errcheck

	// initialize stream and error for the while loop
	var (
		stream                  io.ReadCloser
		streamErr               error
		keepLogging             = true
		bytesSinceLogTimeUpdate = 0
	)

	// get logs from pod, and keep the stream open (follow)
	podLogOptions := &v1.PodLogOptions{
		Follow: true,
	}
	// in case we failed in the middle of streaming logs, we want to start from the last log time
	if lastLogTime > 0 {
		podLogOptions.SinceTime = &metav1.Time{
			Time: time.UnixMilli(lastLogTime),
		}
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
		keepLogging, err = s.streamPodLogs(ctx, runUID, podName, file, stream, &lastLogTime, projectName, &bytesSinceLogTimeUpdate)
		if err != nil {
			// if the pod is still running, it means the logs were rotated, so we need to get a new stream
			// by bailing out
			if !errors.Is(err, common.PodStillRunningError{
				PodName: podName,
			}) {
				s.Logger.WarnWithCtx(ctx,
					"An error occurred while streaming pod logs",
					"err", common.GetErrorStack(err, common.DefaultErrorStackDepth))
			}

			// fatal error, bail out
			// note that when function is returned, a defer function will remove the
			// log collection from (in memory) state file.
			// it ensures us that when log collection monitoring kicks in (it runs periodically)
			// it will ignite the run log collection again.
			return
		}

		// breathe
		// stream pod logs might return fast when there is nothing to read and no error occurred
		time.Sleep(100 * time.Millisecond)
	}

	// remove run from state file
	if err := s.stateManifest.RemoveLogItem(ctx, runUID, projectName); err != nil {
		s.Logger.WarnWithCtx(ctx, "Failed to remove log item from state file")
	}

	s.Logger.DebugWithCtx(ctx,
		"Finished log streaming",
		"projectName", projectName,
		"runUID", runUID,
		"podName", podName)
}

// streamPodLogs streams logs from a pod to a file
func (s *Server) streamPodLogs(ctx context.Context,
	runUID,
	podName string,
	logFile *os.File,
	stream io.ReadCloser,
	logTime *int64,
	projectName string,
	bytesSinceLogTimeUpdate *int) (bool, error) {

	// get a buffer from the pool - so we can share buffers across goroutines
	buf := s.logCollectionBufferPool.Get()
	defer s.logCollectionBufferPool.Put(buf)

	// read from the stream into the buffer
	// this is non-blocking, it will return immediately if there is nothing to read
	numBytesRead, err := stream.Read(buf)
	*logTime = time.Now().UnixMilli()

	if numBytesRead > 0 {

		// write to file
		if _, err := logFile.Write(buf[:numBytesRead]); err != nil {
			s.Logger.WarnWithCtx(ctx,
				"Failed to write pod log to file",
				"err", err.Error(),
				"runUID", runUID)
			return true, errors.Wrap(err, "Failed to write pod log to file")
		}

		// update last log time regularly to mitigate restarts
		*bytesSinceLogTimeUpdate += numBytesRead
		if *bytesSinceLogTimeUpdate >= s.logTimeUpdateBytesInterval {
			*bytesSinceLogTimeUpdate = 0
			if err := s.stateManifest.UpdateLastLogTime(runUID, projectName, *logTime); err != nil {
				return true, errors.Wrap(err, "Failed to update last log time")
			}
		}
	}

	// if error is EOF, it either means the pod is done streaming logs, or the logs in the apiserver were rotated,
	// in such case we need to get a new stream
	if err == io.EOF {

		pod, err := s.kubeClientSet.CoreV1().Pods(s.namespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				s.Logger.DebugWithCtx(ctx,
					"Pod not found, it may have been evicted",
					"runUID", runUID,
					"podName", podName)
				return false, nil
			}
			// some other error occurred
			return false, errors.Wrap(err, "Failed to get pod")
		}
		// if the pod is not running, it is done streaming logs and we can stop
		if pod.Status.Phase != v1.PodRunning {
			s.Logger.DebugWithCtx(ctx, "Pod logs are done streaming", "runUID", runUID)
			return false, nil
		}

		// the pod is still running - exit and without error, so the monitoring loop will continue
		// to stream logs
		s.Logger.DebugWithCtx(ctx, "Received EOF but pod is still running", "runUID", runUID)
		return false, common.PodStillRunningError{
			PodName: podName,
		}
	}

	// other error occurred
	if err != nil {
		return false, errors.Wrap(err, "Failed to read pod logs")
	}

	// nothing happened, continue
	return true, nil
}

// resolveRunLogFilePath returns the path to the pod log file
func (s *Server) resolveRunLogFilePath(projectName, runUID string) string {
	return path.Join(s.baseDir, projectName, runUID)
}

// getLogFilePath returns the path to the run's latest log file
func (s *Server) getLogFilePath(ctx context.Context, runUID, projectName string) (string, error) {
	var logFilePath string
	var retryCount int
	if err := common.RetryUntilSuccessful(5*time.Second, 1*time.Second, func() (bool, error) {
		defer func() {
			retryCount++
		}()

		// verify or wait until project dir exists
		if _, err := os.Stat(filepath.Join(s.baseDir, projectName)); err != nil {
			if os.IsNotExist(err) {
				return true, errors.New("Project directory not found")

				// give v3io-fuse some slack
			} else if strings.Contains(err.Error(), "resource temporarily unavailable") {
				s.Logger.WarnWithCtx(ctx,
					"Project directory is not ready yet (resource temporarily unavailable)",
					"retryCount", retryCount,
					"err", err.Error())
				return true, errors.Wrap(err, "Project directory is not ready yet")
			}
			s.Logger.WarnWithCtx(ctx,
				"Failed to get project directory",
				"retryCount", retryCount,
				"err", err.Error())
			return false, errors.Wrap(err, "Failed to get project directory")
		}

		// get run log file path
		runLogFilePath := s.resolveRunLogFilePath(projectName, runUID)

		if exists, err := common.FileExists(runLogFilePath); err != nil {
			s.Logger.WarnWithCtx(ctx,
				"Failed to get run log file path",
				"retryCount", retryCount,
				"runUID", runUID,
				"projectName", projectName,
				"err", err.Error())
			return false, errors.Wrap(err, "Failed to get project directory")
		} else if !exists {
			s.Logger.WarnWithCtx(ctx,
				"Run log file not found",
				"retryCount", retryCount,
				"runUID", runUID,
				"projectName", projectName)
			return true, errors.New("Run log file not found")
		}

		// found it
		logFilePath = runLogFilePath
		return false, nil
	}); err != nil {
		return "", errors.Wrap(err, "Exhausted getting log file path")
	}

	return logFilePath, nil
}

// readLogsFromFile reads size bytes, starting from offset, from a log file
func (s *Server) readLogsFromFile(ctx context.Context,
	runUID,
	filePath string,
	offset,
	size int64) ([]byte, error) {
	if size == 0 {
		return nil, nil
	}

	fileSize, err := common.GetFileSize(filePath)
	if err != nil {
		return nil, errors.Wrapf(err, "Failed to get file size for run id %s", runUID)
	}

	offset, size = s.validateOffsetAndSize(offset, size, fileSize)
	if size == 0 {
		s.Logger.DebugWithCtx(ctx, "No logs to return", "runUID", runUID)
		return nil, nil
	}

	// open log file for reading
	file, err := os.OpenFile(filePath, os.O_RDONLY, 0600)
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

	// Check the items in the currentState against the items in the state store.
	// If an item is written in the state store but not in the in memory state - call StartLog for it,
	// as the state store is the source of truth
	for range monitoringTicker.C {

		// if there are already log items in progress, call StartLog for each of them
		projectRunUIDsInProgress, err := s.stateManifest.GetItemsInProgress()
		if err == nil {

			logItemsToStart := s.getLogItemsToStart(ctx, projectRunUIDsInProgress)

			errGroup, _ := errgroup.WithContext(ctx)

			for _, logItem := range logItemsToStart {
				logItem := logItem
				errGroup.Go(func() error {
					s.Logger.DebugWithCtx(ctx, "Starting log collection for log item", "runUID", logItem.RunUID)
					if _, err := s.StartLog(ctx, &protologcollector.StartLogRequest{
						RunUID:      logItem.RunUID,
						Selector:    logItem.LabelSelector,
						ProjectName: logItem.Project,
						LastLogTime: logItem.LastLogTime,
					}); err != nil {

						s.Logger.WarnWithCtx(ctx,
							"Failed to start log collection for log item",
							"runUID", logItem.RunUID,
							"project", logItem.Project,
							"err", common.GetErrorStack(err, 10),
						)
						return errors.Wrapf(err, "Failed to start log collection for log item %s", logItem.RunUID)
					}
					return nil
				})
			}

			if err := errGroup.Wait(); err != nil {

				// we don't fail here, there will be a retry in the next iteration
				s.Logger.WarnWithCtx(ctx,
					"Failed to start log collection for some log items",
					"err", common.GetErrorStack(err, 10))
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
	inMemoryInProgress, err := s.currentState.GetItemsInProgress()
	if err != nil {

		// this is just for sanity, as currentState won't return an error
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

// getChunkSuze returns the size of the chunk to read from the log file
func (s *Server) getChunkSize(
	initialOffset,
	totalLogsSize,
	endIndex int64) int64 {

	// we read it all, chunk size is 0
	if totalLogsSize+initialOffset >= endIndex {
		return 0
	}

	// if the size we need to read is bigger than the buffer, use the buffer size
	leftToRead := endIndex - totalLogsSize - initialOffset

	if leftToRead >= int64(s.getLogsBufferSizeBytes) {
		return int64(s.getLogsBufferSizeBytes)
	}
	return leftToRead
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

func (s *Server) deleteRunLogFiles(ctx context.Context, runUID, project string) error {

	// get all files that have the runUID as a prefix
	pattern := path.Join(s.baseDir, project, runUID)
	files, err := filepath.Glob(pattern)
	if err != nil {
		return errors.Wrap(err, "Failed to get log files")
	}

	// delete all matched files
	var failedToDelete []string
	for _, file := range files {
		if err := os.Remove(file); err != nil {

			// don't fail now so the rest of the files will be deleted, just log it
			s.Logger.WarnWithCtx(ctx,
				"Failed to delete log file",
				"file", file,
				"err", common.GetErrorStack(err, 10))
			failedToDelete = append(failedToDelete, file)
		}
	}

	if len(failedToDelete) > 0 {
		return errors.Errorf("Failed to delete log files: %v", failedToDelete)
	}

	return nil
}

func (s *Server) deleteProjectLogs(project string) error {

	// resolve the project logs directory
	projectLogsDir := path.Join(s.baseDir, project)

	// delete the project logs directory (idempotent)
	if err := common.RetryUntilSuccessful(
		1*time.Minute,
		3*time.Second,
		func() (bool, error) {
			if err := os.RemoveAll(projectLogsDir); err != nil {

				// https://stackoverflow.com/a/76921585
				if errors.Is(err, syscall.EBUSY) {

					// try to unmount the directory
					if err := syscall.Unmount(projectLogsDir, 0); err != nil {
						return true, errors.Wrapf(err, "Failed to unmount project logs directory for project %s", project)
					}
				}

				return true, errors.Wrapf(err, "Failed to delete project logs directory for project %s", project)
			}

			// all good, stop retrying
			return false, nil
		}); err != nil {
		return errors.Wrapf(err, "Exhausted deleting project %s directory logs", project)
	}
	return nil
}

func (s *Server) getRunUIDsInProgress(ctx context.Context, inProgressMap *sync.Map, project string) ([]string, error) {
	var runUIDs []string

	inProgressMap.Range(func(projectKey, runUIDsToLogItemsValue interface{}) bool {
		// if a project was provided, only return runUIDs for that project
		if project != "" && project != projectKey {
			return true
		}

		runUIDsToLogItems := runUIDsToLogItemsValue.(*sync.Map)
		runUIDsToLogItems.Range(func(key, value interface{}) bool {
			runUID := key.(string)
			runUIDs = append(runUIDs, runUID)
			return true
		})
		return true
	})

	return runUIDs, nil
}
