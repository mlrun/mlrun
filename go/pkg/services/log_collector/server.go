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
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/mlrun/mlrun/pkg/common"
	"github.com/mlrun/mlrun/pkg/framework"
	"github.com/mlrun/mlrun/proto/build/log_collector"

	"github.com/nuclio/errors"
	"github.com/nuclio/logger"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

type LogCollectorServer struct {
	*framework.AbstractMlrunGRPCServer
	namespace               string
	baseDir                 string
	monitoringInterval      time.Duration
	getLogsInterval         time.Duration
	stateFileUpdateInterval time.Duration
	kubeClientSet           kubernetes.Interface
	state                   *State
	stateLock               sync.Locker
	stateFileLock           sync.Locker
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

	// ensure base dir exists
	if err := common.EnsureDirExists(baseDir, os.ModeDir); err != nil {
		return nil, errors.Wrap(err, "Failed to ensure base dir exists")
	}

	return &LogCollectorServer{
		AbstractMlrunGRPCServer: abstractServer,
		namespace:               namespace,
		baseDir:                 baseDir,
		stateFileUpdateInterval: stateFileUpdateIntervalDuration,
		kubeClientSet:           kubeClientSet,
		stateLock:               &sync.Mutex{},
		stateFileLock:           &sync.Mutex{},
	}, nil
}

func (lcs *LogCollectorServer) OnBeforeStart(ctx context.Context) error {
	lcs.Logger.DebugCtx(ctx, "Initializing Server")

	// load state from file
	state, err := lcs.readStateFile()
	if err != nil {
		return errors.Wrap(err, "Failed to load state file")
	}
	lcs.state = state

	// start state updating goroutine
	go lcs.updateStateFile(ctx, lcs.state)

	// if there are items in the stateFile, call StartLog for each of them
	for runId, logItem := range state.InProgress {
		lcs.Logger.DebugWithCtx(ctx, "Starting log collection for log item", "runId", runId)
		if _, err := lcs.StartLog(ctx, &log_collector.StartLogRequest{
			RunId:    runId,
			Selector: logItem.LabelSelector,
		}); err != nil {
			lcs.Logger.WarnWithCtx(ctx, "Failed to start log collection for log item", "runId", runId)
			continue
		}
	}

	return nil
}

func (lcs *LogCollectorServer) RegisterRoutes(ctx context.Context) {
	lcs.AbstractMlrunGRPCServer.RegisterRoutes(ctx)
	log_collector.RegisterLogCollectorServer(lcs.Server, lcs)
}

// StartLog writes the log item info to the state file, gets the pod using the label selector,
// triggers `monitorPod` and `streamLogs` goroutines.
func (lcs *LogCollectorServer) StartLog(ctx context.Context, request *log_collector.StartLogRequest) (*log_collector.StartLogResponse, error) {

	lcs.Logger.DebugWithCtx(ctx, "Received Start Log request", "request", request)

	// write log item in progress to file
	if err := lcs.addItemToInProgress(ctx, request.RunId, request.Selector); err != nil {
		err := errors.Wrapf(err, "Failed to add run id %s to state file", request.RunId)
		return &log_collector.StartLogResponse{
			Success: false,
			Error:   err.Error(),
		}, err
	}

	var pods *v1.PodList
	var err error

	// list pods using label selector until a pod is found
	for len(pods.Items) == 0 {
		// TODO: add timeout to this loop

		pods, err = lcs.kubeClientSet.CoreV1().Pods(lcs.namespace).List(ctx, metav1.ListOptions{
			LabelSelector: request.Selector,
		})
		if err != nil {
			err := errors.Wrapf(err, "Failed to list pods for run id %s", request.RunId)
			return &log_collector.StartLogResponse{
				Success: false,
				Error:   err.Error(),
			}, err
		}
	}

	// found a pod. for now, we only assume each run has a single pod.
	pod := pods.Items[0]

	// TODO: create a child context before calling goroutines, so it won't be canceled

	// stream logs to file
	go lcs.streamPodLogs(ctx, request.RunId, pod.Name)

	return &log_collector.StartLogResponse{
		Success: true,
	}, nil
}

// GetLog returns the log file contents of length size from an offset, for a given run id
func (lcs *LogCollectorServer) GetLog(ctx context.Context, request *log_collector.GetLogRequest) (*log_collector.GetLogResponse, error) {
	lcs.Logger.DebugWithCtx(ctx, "Received Get Log request", "request", request)

	// get log file path
	filePath, err := lcs.getLogFilePath(request.RunId)
	if err != nil {
		err := errors.Wrapf(err, "Failed to get log file path for run id %s", request.RunId)
		return &log_collector.GetLogResponse{
			Success: false,
			Error:   err.Error(),
		}, err
	}

	// open log file
	file, err := os.Open(filePath)
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

	offset, size := lcs.validateOffsetAndSize(request.Offset, request.Size, uint64(fileInfo.Size()))
	if size == 0 {
		lcs.Logger.DebugWithCtx(ctx, "No logs to return", "run_id", request.RunId)
		return &log_collector.GetLogResponse{
			Success: true,
			Log:     []byte{},
		}, nil
	}

	// read size bytes from offset
	buffer := make([]byte, size)
	_, err = file.ReadAt(buffer, int64(offset))
	if err != nil {

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
func (lcs *LogCollectorServer) streamPodLogs(ctx context.Context, runId, podName string) {

	// create a log file to the pod
	filePath := lcs.resolvePodLogFilePath(runId, podName)
	if err := common.EnsureFileExists(filePath); err != nil {
		lcs.Logger.ErrorWithCtx(ctx,
			"Failed to ensure log file",
			"runId", runId,
			"filePath", filePath)
		return
	}

	// get logs from pod, and keep the stream open (follow)
	podLogOptions := &v1.PodLogOptions{
		Follow: true,
	}
	restClientRequest := lcs.kubeClientSet.CoreV1().Pods(lcs.namespace).GetLogs(podName, podLogOptions)

	// initialize error for the while loop
	streamErr := errors.New("Stream error")
	var stream io.ReadCloser

	// stream logs - retry if failed
	for streamErr != nil {
		stream, streamErr = restClientRequest.Stream(ctx)
		if streamErr != nil {
			lcs.Logger.WarnWithCtx(ctx, "Failed to get pod log read/closer", "runId", runId)
		}
		time.Sleep(1 * time.Second)
	}
	defer stream.Close() // nolint: errcheck

	for {
		buf := make([]byte, 1024)

		// This is blocking until there is something to read
		numBytes, err := stream.Read(buf)

		// nothing read, continue
		if numBytes == 0 {
			continue
		}

		// if error is EOF, the pod is done streaming logs (deleted/completed/failed)
		if err == io.EOF {
			lcs.Logger.DebugWithCtx(ctx, "Pod logs are done streaming", "runId", runId)
			break
		}

		// if error is not EOF, log it and continue
		if err != nil {
			lcs.Logger.WarnWithCtx(ctx, "Failed to read pod log",
				"error",
				err, "runId", runId)
			continue
		}

		// write log contents to file
		if err := common.WriteToFile(ctx, lcs.Logger, filePath, buf[:numBytes], true); err != nil {
			lcs.Logger.ErrorWithCtx(ctx,
				"Failed to write log contents to file",
				"runId", runId,
				"podName", podName)
		}
	}

	lcs.Logger.DebugWithCtx(ctx,
		"Removing item from state file",
		"runId", runId,
		"podName", podName)

	// remove run id from state file
	if err := lcs.removeItemFromInProgress(runId); err != nil {
		lcs.Logger.WarnWithCtx(ctx, "Failed to remove run id from state file")
	}
}

// addItemToInProgress adds an item to the `in_progress` list in the state file
func (lcs *LogCollectorServer) addItemToInProgress(ctx context.Context, runId string, selector string) error {

	state := lcs.getState()

	logItem := LogItem{
		RunId:         runId,
		LabelSelector: selector,
	}

	if existingItem, exists := state.InProgress[runId]; exists {
		lcs.Logger.DebugWithCtx(ctx,
			"Item already exists in state file. Overwriting label selector",
			"runId", runId,
			"existingItem", existingItem)
		// TODO: notify the goroutines to stop and restart them with the new selector
	}

	state.InProgress[runId] = logItem

	lcs.setState(state)

	return nil
}

// removeItemFromInProgress removes an item from the `in_progress` list in the state file
func (lcs *LogCollectorServer) removeItemFromInProgress(runId string) error {

	// get state file
	state := lcs.getState()

	// remove run id item from state
	delete(state.InProgress, runId)

	lcs.setState(state)

	return nil
}

func (lcs *LogCollectorServer) getState() *State {
	lcs.stateLock.Lock()
	defer lcs.stateLock.Unlock()

	return lcs.state

}

func (lcs *LogCollectorServer) setState(state *State) {
	lcs.stateLock.Lock()
	defer lcs.stateLock.Unlock()

	lcs.state = state
}

// getState returns the state file
func (lcs *LogCollectorServer) readStateFile() (*State, error) {

	// get lock
	lcs.stateFileLock.Lock()
	defer lcs.stateFileLock.Unlock()

	// read file
	stateFilePath := lcs.resolveStateFilePath()
	if err := common.EnsureFileExists(stateFilePath); err != nil {
		return nil, errors.Wrap(err, "Failed to ensure state file exists")
	}
	stateFileBytes, err := os.ReadFile(stateFilePath)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to read stateFile")
	}

	state := &State{}

	if len(stateFileBytes) == 0 {

		// if file is empty, return the empty state instance
		return &State{
			InProgress: map[string]LogItem{},
		}, nil
	}

	// unmarshal
	if err := json.Unmarshal(stateFileBytes, state); err != nil {
		return nil, errors.Wrap(err, "Failed to unmarshal state file")
	}

	return state, nil
}

// writeStateToFile writes the state to file
func (lcs *LogCollectorServer) writeStateToFile(ctx context.Context, state *State) error {

	// marshal state file
	encodedState, err := json.Marshal(state)
	if err != nil {
		return errors.Wrap(err, "Failed to encode state file")
	}

	// get lock, unlock later
	lcs.stateFileLock.Lock()
	defer lcs.stateFileLock.Unlock()

	// write to file
	stateFilePath := lcs.resolveStateFilePath()
	return common.WriteToFile(ctx, lcs.Logger, stateFilePath, encodedState, false)
}

// resolvePodLogFilePath returns the path to the pod log file
func (lcs *LogCollectorServer) resolvePodLogFilePath(runId, podName string) string {
	return path.Join(lcs.baseDir, "logs", fmt.Sprintf("%s_%s.log", runId, podName))
}

func (lcs *LogCollectorServer) resolveStateFilePath() string {
	return path.Join(lcs.baseDir, "state.json")
}

func (lcs *LogCollectorServer) getLogFilePath(runId string) (string, error) {

	logFilePath := ""
	var latestModTime time.Time

	// list all files in base directory
	err := filepath.Walk(lcs.baseDir, func(path string, info os.FileInfo, err error) error {
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

func (lcs *LogCollectorServer) validateOffsetAndSize(offset uint64, size uint64, fileSize uint64) (uint64, uint64) {

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

// updateStateFile periodically checks if changes were made to the state, and updates them in the state file
func (lcs *LogCollectorServer) updateStateFile(ctx context.Context, state *State) {

	for {

		// get state file
		currentState := lcs.getState()

		// if state changed, write it to file
		if !reflect.DeepEqual(currentState, state) {

			state = currentState

			// write state file
			if err := lcs.writeStateToFile(ctx, state); err != nil {
				lcs.Logger.ErrorWithCtx(ctx, "Failed to write state file", "err", err)
				return
			}
		}

		time.Sleep(lcs.stateFileUpdateInterval)
	}
}
