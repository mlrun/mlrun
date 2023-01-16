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
	"runtime/debug"
	"strings"
	"time"

	"github.com/mlrun/mlrun/pkg/common"
	mlruncontext "github.com/mlrun/mlrun/pkg/context"
	"github.com/mlrun/mlrun/pkg/framework"
	"github.com/mlrun/mlrun/pkg/services/logcollector/statestore"
	"github.com/mlrun/mlrun/pkg/services/logcollector/statestore/factory"
	protologcollector "github.com/mlrun/mlrun/proto/build/log_collector"

	"github.com/nuclio/errors"
	"github.com/nuclio/logger"
	"github.com/oxtoacart/bpool"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

type LogCollectorServer struct {
	*framework.AbstractMlrunGRPCServer
	namespace          string
	baseDir            string
	kubeClientSet      kubernetes.Interface
	stateStore         statestore.StateStore
	bufferPool         *bpool.BytePool
	readLogWaitTime    time.Duration
	monitoringInterval time.Duration
	inMemoryState      statestore.StateStore
}

func NewLogCollectorServer(logger logger.Logger,
	namespace,
	baseDir,
	kubeconfigPath,
	stateFileUpdateInterval,
	readLogWaitTime,
	monitoringInterval string,
	bufferPoolSize,
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
		return nil, errors.Wrap(err, "Failed to state file updating interval")
	}
	readLogTimeoutDuration, err := time.ParseDuration(readLogWaitTime)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to parse read log wait time duration")
	}
	monitoringIntervalDuration, err := time.ParseDuration(monitoringInterval)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to parse monitoring interval")
	}

	stateStore, err := factory.CreateStateStore(
		statestore.StateStoreTypeFile,
		&statestore.Config{
			Logger:                  logger,
			StateFileUpdateInterval: stateFileUpdateIntervalDuration,
			BaseDir:                 baseDir,
		},
	)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to create state store")
	}

	inMemoryState, err := factory.CreateStateStore(statestore.StateStoreTypeInMemory, &statestore.Config{})
	if err != nil {
		return nil, errors.Wrap(err, "Failed to create in-memory state")
	}

	// ensure base dir exists
	if err := common.EnsureDirExists(baseDir, os.ModeDir); err != nil {
		return nil, errors.Wrap(err, "Failed to ensure base dir exists")
	}

	// create a byte buffer pool - a pool of size `bufferPoolSize`, where each buffer is of size `bufferSizeBytes`
	bufferPool := bpool.NewBytePool(bufferPoolSize, bufferSizeBytes)

	return &LogCollectorServer{
		AbstractMlrunGRPCServer: abstractServer,
		namespace:               namespace,
		baseDir:                 baseDir,
		stateStore:              stateStore,
		kubeClientSet:           kubeClientSet,
		readLogWaitTime:         readLogTimeoutDuration,
		monitoringInterval:      monitoringIntervalDuration,
		bufferPool:              bufferPool,
		inMemoryState:           inMemoryState,
	}, nil
}

func (s *LogCollectorServer) OnBeforeStart(ctx context.Context) error {
	s.Logger.DebugCtx(ctx, "Initializing Server")

	// start state updating goroutine
	go s.stateStore.Initialize(ctx)

	// start logging monitor
	go s.monitorLogCollection(ctx)

	// if there are already log items in progress, call StartLog for each of them
	logItemsInProgress, err := s.stateStore.GetItemsInProgress()
	if err == nil {
		logItemsInProgress.Range(func(key, value any) bool {
			runUID, ok := key.(string)
			if !ok {
				s.Logger.WarnWithCtx(ctx, "Failed to convert runUID key to string")
			}
			logItem, ok := value.(statestore.LogItem)
			if !ok {
				s.Logger.WarnWithCtx(ctx, "Failed to convert in progress item to logItem")
			}

			s.Logger.DebugWithCtx(ctx, "Starting log collection for log item", "runUID", runUID)
			if _, err := s.StartLog(ctx, &protologcollector.StartLogRequest{
				RunUID:   runUID,
				Selector: logItem.LabelSelector,
			}); err != nil {
				// we don't fail here, as there might be other items to start log for, just log it
				s.Logger.WarnWithCtx(ctx, "Failed to start log collection for log item", "runUID", runUID)
			}
			return true
		})
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

	// check if the requested run uid's logs are already being collected
	itemsInProgress, err := s.inMemoryState.GetItemsInProgress()
	if err != nil {
		err := errors.Wrap(err, "Failed to get items in progress from in memory state")
		return &protologcollector.StartLogResponse{
			Success: false,
			Error:   err.Error(),
		}, err
	}
	if _, running := itemsInProgress.Load(request.RunUID); running {
		s.Logger.DebugWithCtx(ctx, "Logs are already being collected for this run uid", "runUID", request.RunUID)
		return &protologcollector.StartLogResponse{
			Success: true,
		}, nil
	}

	var pods *v1.PodList

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
				s.Logger.WarnWithCtx(ctx, "Failed to list pods, retrying", "runUID", request.RunUID)
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

	// write log item in progress to state store
	if err := s.stateStore.AddLogItem(ctx, request.RunUID, request.Selector); err != nil {
		err := errors.Wrapf(err, "Failed to add run id %s to state file", request.RunUID)
		return &protologcollector.StartLogResponse{
			Success: false,
			Error:   err.Error(),
		}, err
	}

	// create a child context before calling goroutines, so it won't be canceled
	logStreamCtx, cancelCtxFunc := mlruncontext.NewDetachedWithCancel(ctx)
	startedStreamingGoroutine := make(chan bool, 1)

	// stream logs to file
	go s.startLogStreaming(logStreamCtx, request.RunUID, pod.Name, startedStreamingGoroutine, cancelCtxFunc)

	// add Item to in-memory state
	if err := s.inMemoryState.AddLogItem(ctx, request.RunUID, request.Selector); err != nil {
		err := errors.Wrapf(err, "Failed to add run id %s to in memory state", request.RunUID)
		return &protologcollector.StartLogResponse{
			Success: false,
			Error:   err.Error(),
		}, err
	}

	<-startedStreamingGoroutine

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
func (s *LogCollectorServer) startLogStreaming(ctx context.Context,
	runUID,
	podName string,
	startedStreamingGoroutine chan bool,
	cancelCtxFunc context.CancelFunc) {

	// in case of a panic, remove this goroutine from the in-memory state, so the
	// monitoring loop will start logging again for this runUID.
	defer func() {
		if err := recover(); err != nil {
			callStack := debug.Stack()
			s.Logger.ErrorWithCtx(ctx, "Panic caught while creating function",
				"err", err,
				"stack", string(callStack))

			// remove this goroutine from in-memory state
			if err := s.inMemoryState.RemoveLogItem(runUID); err != nil {
				s.Logger.WarnWithCtx(ctx, "Failed to remove item from in memory state")
			}

			// cancel all other goroutines spawned from this one
			cancelCtxFunc()
		}
	}()

	// signal "main" function that goroutine is up
	startedStreamingGoroutine <- true

	// create a log file to the pod
	logFilePath := s.resolvePodLogFilePath(runUID, podName)
	if err := common.EnsureFileExists(logFilePath); err != nil {
		s.Logger.ErrorWithCtx(ctx,
			"Failed to ensure log file",
			"runUID", runUID,
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
		s.Logger.WarnWithCtx(ctx, "Failed to get pod log stream read/closer, retrying", "runUID", runUID, "err", streamErr.Error())
		time.Sleep(1 * time.Second)
	}
	defer stream.Close() // nolint: errcheck

	for {
		keepLogging, err := s.streamPodLogs(ctx, runUID, logFilePath, stream)
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
		"runUID", runUID,
		"podName", podName)

	// remove run from state file
	if err := s.stateStore.RemoveLogItem(runUID); err != nil {
		s.Logger.WarnWithCtx(ctx, "Failed to remove log item from state file")
	}
}

func (s *LogCollectorServer) streamPodLogs(ctx context.Context,
	runUID,
	logFilePath string,
	stream io.ReadCloser) (bool, error) {

	// create a reader from the stream, to allow peeking into it
	streamReader := bufio.NewReader(stream)

	// wait for the stream ro have logs before reading them
	if !s.hasLogs(ctx, runUID, streamReader) {
		s.Logger.WarnWithCtx(ctx, "Stream doesn't have logs or context has been canceled", "runUID", runUID)
		return false, nil
	}

	// open log file in read/write and append, to allow reading the logs while we write more logs to it
	openFlags := os.O_RDWR | os.O_APPEND
	file, err := os.OpenFile(logFilePath, openFlags, 0644)
	if err != nil {
		s.Logger.WarnWithCtx(ctx, "Failed to open file", "err", err, "logFilePath", logFilePath)
		return true, err
	}
	defer file.Close() // nolint: errcheck

	// spin a goroutine that will unblock `CopyBuffer` if context is dead
	copyBufferDone := make(chan struct{}, 1)
	go s.cancelOnContext(ctx, file, copyBufferDone)

	// get a buffer from the pool - so we can share buffers across goroutines
	buf := s.bufferPool.Get()
	defer s.bufferPool.Put(buf)

	// copy the stream into the file using the buffer, which allows us to control the size read from the file.
	// this is blocking until there is something to read
	numBytesWritten, err := io.CopyBuffer(file, streamReader, buf)

	// signal goroutine to exit
	close(copyBufferDone)

	// if error is EOF, the pod is done streaming logs (deleted/completed/failed)
	if err == io.EOF {
		s.Logger.DebugWithCtx(ctx, "Pod logs are done streaming", "runUID", runUID)
		return false, nil
	}

	// nothing read, continue
	if numBytesWritten == 0 {
		return true, nil
	}

	// if error is not EOF, log it and continue
	if err != nil {
		s.Logger.WarnWithCtx(ctx, "Failed to read pod log",
			"err", err.Error(),
			"runUID", runUID)
		return true, errors.Wrap(err, "Failed to read pod logs")
	}

	// sanity
	return true, nil
}

// resolvePodLogFilePath returns the path to the pod log file
func (s *LogCollectorServer) resolvePodLogFilePath(runUID, podName string) string {
	return path.Join(s.baseDir, "logs", fmt.Sprintf("%s_%s.log", runUID, podName))
}

func (s *LogCollectorServer) hasLogs(ctx context.Context, runUID string, streamReader *bufio.Reader) bool {

	// peek into the stream, and wait until there is something to read from it
	// or until context is canceled
	for {
		select {
		case <-time.After(s.readLogWaitTime):
			peekBuf, err := streamReader.Peek(1)

			// if there is something to read, return true
			// if error is EOF, the pod has logs but not new ones
			if err == io.EOF || len(peekBuf) > 0 {
				return true
			}
			if err != nil {
				s.Logger.WarnWithCtx(ctx, "Failed to peek into stream", "runUID", runUID, "err", err.Error())
			}
		case <-ctx.Done():
			s.Logger.DebugWithCtx(ctx, "Context was canceled, stopping waiting for pod log stream", "runUID", runUID)
			return false
		}
		time.Sleep(s.readLogWaitTime)
	}
}

func (s *LogCollectorServer) cancelOnContext(ctx context.Context, file *os.File, copyBufferDone chan struct{}) {
	select {
	case <-ctx.Done():

		// context is dead, so we close the file so `CopyBuffer` won't block and fail
		file.Close() // nolint: errcheck
	case <-copyBufferDone:

		// `CopyBuffer doesn't block anymore, we can stop the goroutine
		return
	}
}

func (s *LogCollectorServer) getLogFilePath(runUID string) (string, error) {

	logFilePath := ""
	var latestModTime time.Time

	// list all files in base directory
	err := filepath.Walk(s.baseDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// if file name starts with run id, it's a log file
		if strings.HasPrefix(info.Name(), runUID) {

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
		return "", errors.Errorf("Failed to find log file for run id %s", runUID)
	}

	return logFilePath, nil
}

func (s *LogCollectorServer) readLogsFromFile(ctx context.Context,
	runUID,
	filePath string,
	offset uint64,
	size int64) ([]byte, error) {

	// open log file for reading
	file, err := os.OpenFile(filePath, os.O_RDONLY, 0644)
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

func (s *LogCollectorServer) monitorLogCollection(ctx context.Context) {

	s.Logger.DebugWithCtx(ctx,
		"Monitoring log streaming goroutines periodically",
		"monitoringInterval", s.monitoringInterval)

	monitoringTicker := time.NewTicker(s.monitoringInterval)

	// Check the items in the inMemoryState against the items in the state store.
	// If an item is written in the state store but not in the in memory state - call StartLog for it,
	// as the state store is the source of truth
	for range monitoringTicker.C {

		// if there are already log items in progress, call StartLog for each of them
		logItemsInProgress, err := s.stateStore.GetItemsInProgress()
		if err == nil {
			logItemsInProgress.Range(func(key, value any) bool {
				runUID, ok := key.(string)
				if !ok {
					s.Logger.WarnWithCtx(ctx, "Failed to convert runUID key to string")
				}
				logItem, ok := value.(statestore.LogItem)
				if !ok {
					s.Logger.WarnWithCtx(ctx, "Failed to convert in progress item to logItem")
				}

				inMemoryInProgress, err := s.inMemoryState.GetItemsInProgress()
				if err != nil {
					s.Logger.WarnWithCtx(ctx, "Failed to get in progress items from in memory state", "err", err)
					return true
				}

				// check if the log streaming is already running for this runUID
				if _, running := inMemoryInProgress.Load(runUID); !running {

					s.Logger.DebugWithCtx(ctx, "Starting log collection for log item", "runUID", runUID)
					if _, err := s.StartLog(ctx, &protologcollector.StartLogRequest{
						RunUID:   runUID,
						Selector: logItem.LabelSelector,
					}); err != nil {

						// we don't fail here, as there might be other items to start log for, just log it
						s.Logger.WarnWithCtx(ctx, "Failed to start log collection for log item", "runUID", runUID)
					}
				}

				return true
			})
		} else {

			// don't fail because we still need the server to run
			s.Logger.WarnWithCtx(ctx, "Failed to get log items in progress")
		}
	}
}
