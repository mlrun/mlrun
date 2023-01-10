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
	"bytes"
	"context"
	"encoding/json"
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
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/kubernetes"
)

type LogCollectorServer struct {
	*framework.AbstractMlrunGRPCServer
	namespace          string
	baseDir            string
	monitoringInterval time.Duration
	getLogsInterval    time.Duration
	stateFilePath      string
	kubeClientSet      kubernetes.Interface
	stateFileLock      sync.Locker
}

func NewLogCollectorServer(logger logger.Logger,
	namespace,
	baseDir,
	stateFilePath,
	kubeconfigPath,
	monitoringInterval,
	getLogsInterval string) (*LogCollectorServer, error) {
	abstractServer, err := framework.NewAbstractMlrunGRPCServer(logger, nil)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to create abstract server")
	}

	// initialize kubernetes client
	restConfig, err := common.GetClientConfig(kubeconfigPath)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to get client configuration")
	}
	kubeClientSet, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to create kubernetes client set")
	}

	// parse interval durations
	monitoringIntervalDuration, err := time.ParseDuration(monitoringInterval)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to parse monitoring interval")
	}
	getLogsIntervalDuration, err := time.ParseDuration(getLogsInterval)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to parse monitoring interval")
	}

	return &LogCollectorServer{
		AbstractMlrunGRPCServer: abstractServer,
		namespace:               namespace,
		baseDir:                 baseDir,
		monitoringInterval:      monitoringIntervalDuration,
		getLogsInterval:         getLogsIntervalDuration,
		stateFilePath:           stateFilePath,
		kubeClientSet:           kubeClientSet,
		stateFileLock:           &sync.Mutex{},
	}, nil
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
		pods, err = lcs.kubeClientSet.CoreV1().Pods(lcs.namespace).List(ctx, metav1.ListOptions{
			LabelSelector: labels.SelectorFromSet(request.Selector).String(),
		})
		if err != nil {
			err := errors.Wrapf(err, "Failed to list pods for run id %s", request.RunId)
			return &log_collector.StartLogResponse{
				Success: false,
				Error:   err.Error(),
			}, err
		}
	}

	// found a pod
	pod := pods.Items[0]

	// TODO: create a child context before calling goroutines?

	//start monitoring it
	stopChan := make(chan struct{}, 1)
	go lcs.monitorPodState(ctx, pod.Name, stopChan)

	// stream logs to file
	go lcs.streamPodLogs(ctx, request.RunId, pod.Name, stopChan)

	return &log_collector.StartLogResponse{
		Success: true,
	}, nil
}

// GetLog returns the log file contents of length size from an offset, for a given run id
func (lcs *LogCollectorServer) GetLog(ctx context.Context, request *log_collector.GetLogRequest) (*log_collector.GetLogResponse, error) {
	lcs.Logger.DebugWithCtx(ctx, "Received Get Log request", "request", request)

	// get log file path
	filePath, err := lcs.getLogFilePath(ctx, request.RunId)
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
	if offset == 0 || size == 0 {
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

// monitorPodState monitors the pod, and stops log streaming if the pod reached completed or failed state
func (lcs *LogCollectorServer) monitorPodState(ctx context.Context, podName string, stopChan chan struct{}) {

	lcs.Logger.DebugWithCtx(ctx, "Starting pod state monitoring", "podName", podName)

	for {

		// get pod
		pod, err := lcs.kubeClientSet.CoreV1().Pods(lcs.namespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			lcs.Logger.WarnWithCtx(ctx, "Failed to get pod", "podName", podName)
		}

		// check pod state. if completed / failed - invoke stopChan and return
		if pod.Status.Phase == v1.PodSucceeded || pod.Status.Phase == v1.PodFailed {
			lcs.Logger.DebugWithCtx(ctx, "Stopping pod state monitoring", "podName", podName)
			stopChan <- struct{}{}
			return
		}

		time.Sleep(lcs.monitoringInterval)
	}
}

// streamPodLogs streams logs from a pod and writes them into a file
func (lcs *LogCollectorServer) streamPodLogs(ctx context.Context, runId, podName string, stopChan chan struct{}) {
	var prevLogTime *metav1.Time

	for {
		select {
		case <-time.After(lcs.getLogsInterval):

			// get logs from pod.
			// since we're in appending to the file, get the logs since the last time we got them
			podLogOptions := &v1.PodLogOptions{
				SinceTime: prevLogTime,
			}
			*prevLogTime = metav1.Now()
			restClientRequest := lcs.kubeClientSet.CoreV1().Pods(lcs.namespace).GetLogs(podName, podLogOptions)

			// stream logs
			restReadCloser, err := restClientRequest.Stream(ctx)
			if err != nil {
				lcs.Logger.WarnWithCtx(ctx, "Failed to get pod log read/closer", "runId", runId)
				//return "", errors.Wrap(err, "Failed to get pod log read/closer", "runId", runId)
			}

			buf := new(bytes.Buffer)
			_, err = io.Copy(buf, restReadCloser)
			restReadCloser.Close() // nolint: errcheck
			if err != nil {
				lcs.Logger.WarnWithCtx(ctx, "Failed to copy information from log stream to buffer", "runId", runId)
				//return "", errors.Wrap(err, "Failed to read pod logs", "runId", runId)
			}

			// write log contents to file
			filePath := lcs.resolvePodLogFilePath(runId, podName)
			if err := lcs.writeToFile(ctx, filePath, buf.Bytes(), true); err != nil {
				lcs.Logger.ErrorWithCtx(ctx,
					"Failed to write log contents to file",
					"runId", runId,
					"podName", podName)
			}

		case <-stopChan:
			lcs.Logger.DebugWithCtx(ctx,
				"Removing item from state file",
				"runId", runId,
				"podName", podName)

			// remove run id from state file
			if err := lcs.removeItemFromInProgress(ctx, runId); err != nil {
				lcs.Logger.WarnWithCtx(ctx, "Failed to remove run id from state file")
			}

			return
		}
	}
}

// addItemToInProgress adds an item to the `in_progress` list in the state file
func (lcs *LogCollectorServer) addItemToInProgress(ctx context.Context, runId string, selector map[string]string) error {

	stateFile, err := lcs.getStateFile()
	if err != nil {
		return errors.Wrap(err, "Failed to get state file")
	}

	loggedItem := LoggedItem{
		RunId:         runId,
		LabelSelector: selector,
	}

	stateFile.InProgress = append(stateFile.InProgress, loggedItem)

	if err := lcs.writeStateFile(ctx, stateFile); err != nil {
		return errors.Wrap(err, "Failed to write state file")
	}

	return nil
}

// removeItemFromInProgress removes an item from the `in_progress` list in the state file
func (lcs *LogCollectorServer) removeItemFromInProgress(ctx context.Context, runId string) error {

	// get state file
	stateFile, err := lcs.getStateFile()
	if err != nil {
		return errors.Wrap(err, "Failed to get state file")
	}

	// find and remove run id item
	indexToDelete := 0
	for i, loggedItem := range stateFile.InProgress {
		if loggedItem.RunId == runId {
			indexToDelete = i
		}
	}

	// remove item from in progress slice
	stateFile.InProgress[indexToDelete] = stateFile.InProgress[len(stateFile.InProgress)-1]
	stateFile.InProgress = stateFile.InProgress[:len(stateFile.InProgress)-1]

	// write state file back
	if err := lcs.writeStateFile(ctx, stateFile); err != nil {
		return errors.Wrap(err, "Failed to write state file")
	}

	return nil
}

// getStateFile returns the state file
func (lcs *LogCollectorServer) getStateFile() (*StateFile, error) {

	// get lock
	lcs.stateFileLock.Lock()
	defer lcs.stateFileLock.Unlock()

	// read file
	// TODO: what if file doesn't exist?
	stateFileBytes, err := os.ReadFile(lcs.stateFilePath)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to read stateFile")
	}

	stateFile := &StateFile{}

	if len(stateFileBytes) == 0 {

		// if file is empty, return the empty state file instance
		return stateFile, nil
	}

	// unmarshal
	if err := json.Unmarshal(stateFileBytes, stateFile); err != nil {
		return nil, errors.Wrap(err, "Failed to unmarshal state file")
	}

	return stateFile, nil
}

// writeStateFile writes the state file
func (lcs *LogCollectorServer) writeStateFile(ctx context.Context, stateFile *StateFile) error {

	// marshal state file
	encodedStateFile, err := json.Marshal(stateFile)
	if err != nil {
		return errors.Wrap(err, "Failed to encode state file")
	}

	// get lock, unlock later
	lcs.stateFileLock.Lock()
	defer lcs.stateFileLock.Unlock()

	// write to file
	return lcs.writeToFile(ctx, lcs.stateFilePath, encodedStateFile, false)
}

// writeToFile writes the given bytes to the given file path
func (lcs *LogCollectorServer) writeToFile(ctx context.Context, filePath string, content []byte, append bool) error {

	// this flag enables us to create the file if it doesn't exist
	openFlags := os.O_CREATE
	if append {
		openFlags = os.O_APPEND | os.O_CREATE
	}

	// open file
	file, err := os.OpenFile(filePath, openFlags, 0600)
	if err != nil {
		return errors.Wrapf(err, "Failed to open file - %s", filePath)
	}

	defer file.Close() // nolint: errcheck

	lcs.Logger.DebugWithCtx(ctx, "Writing log contents to file", "filePath", filePath)
	if _, err := file.Write(content); err != nil {
		return errors.Wrapf(err, "Failed to write log contents to file - %s", filePath)
	}

	return nil
}

// resolvePodLogFilePath returns the path to the pod log file
func (lcs *LogCollectorServer) resolvePodLogFilePath(runId, podName string) string {
	return path.Join(lcs.baseDir, fmt.Sprintf("%s_%s", runId, podName))
}

func (lcs *LogCollectorServer) getLogFilePath(ctx context.Context, runId string) (string, error) {

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
