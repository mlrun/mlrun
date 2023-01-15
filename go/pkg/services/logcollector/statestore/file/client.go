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

package file

import (
	"context"
	"encoding/json"
	"os"
	"path"
	"reflect"
	"sync"
	"time"

	"github.com/mlrun/mlrun/pkg/common"
	"github.com/mlrun/mlrun/pkg/services/logcollector/statestore"

	"github.com/nuclio/errors"
	"github.com/nuclio/logger"
)

type FileStateStore struct {
	state                   *statestore.State
	logger                  logger.Logger
	stateFilePath           string
	stateFileUpdateInterval time.Duration
	lock                    sync.Locker
}

func NewFileStateStore(logger logger.Logger, baseDirPath string, stateFileUpdateInterval time.Duration) *FileStateStore {
	return &FileStateStore{
		state: &statestore.State{
			InProgress: &sync.Map{},
		},
		logger:                  logger.GetChild("filestatestore"),
		stateFilePath:           path.Join(baseDirPath, "state.json"),
		stateFileUpdateInterval: stateFileUpdateInterval,
		lock:                    &sync.Mutex{},
	}
}

func (f *FileStateStore) Initialize(ctx context.Context) {

	// spawn a goroutine that will update the state file periodically
	go f.stateFileUpdateLoop(ctx)
}

func (f *FileStateStore) AddLogItem(ctx context.Context, runUID, selector string) error {
	logItem := statestore.LogItem{
		RunUID:        runUID,
		LabelSelector: selector,
	}

	if existingItem, exists := f.state.InProgress.Load(runUID); exists {
		f.logger.DebugWithCtx(ctx,
			"Item already exists in state file. Overwriting label selector",
			"runUID", runUID,
			"existingItem", existingItem)
	}

	f.state.InProgress.Store(logItem.RunUID, logItem)
	return nil
}

func (f *FileStateStore) RemoveLogItem(runUID string) error {
	f.state.InProgress.Delete(runUID)
	return nil
}

// WriteState writes the state to file, used mainly for testing
func (f *FileStateStore) WriteState(state *statestore.State) error {
	return f.writeStateToFile(state)
}

func (f *FileStateStore) GetItemsInProgress() (*sync.Map, error) {
	state, err := f.readStateFile()
	if err != nil {
		return nil, errors.Wrap(err, "Failed to read state file")
	}

	// set the state in the file state store
	f.lock.Lock()
	defer f.lock.Unlock()
	f.state = state

	return f.state.InProgress, nil
}

func (f *FileStateStore) GetState() *statestore.State {
	return f.state
}

func (f *FileStateStore) stateFileUpdateLoop(ctx context.Context) {

	prevState := *f.state

	for {

		// get state
		currentState := *f.state

		// if state changed, write it to file
		if !reflect.DeepEqual(currentState, prevState) {

			prevState = currentState

			// write state file
			if err := f.writeStateToFile(&currentState); err != nil {
				f.logger.ErrorWithCtx(ctx, "Failed to write state file", "err", err)
				return
			}
		}

		time.Sleep(f.stateFileUpdateInterval)
	}
}

// writeStateToFile writes the state to file
func (f *FileStateStore) writeStateToFile(state *statestore.State) error {

	// marshal state file
	encodedState, err := json.Marshal(state)
	if err != nil {
		return errors.Wrap(err, "Failed to encode state file")
	}

	// get lock, unlock later
	f.lock.Lock()
	defer f.lock.Unlock()

	// write to file
	return common.WriteToFile(f.stateFilePath, encodedState, false)
}

// readStateFile reads the state from the file
func (f *FileStateStore) readStateFile() (*statestore.State, error) {

	// get lock so file won't be updated while reading
	f.lock.Lock()
	defer f.lock.Unlock()

	// read file
	if err := common.EnsureFileExists(f.stateFilePath); err != nil {
		return nil, errors.Wrap(err, "Failed to ensure state file exists")
	}
	stateFileBytes, err := os.ReadFile(f.stateFilePath)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to read stateFile")
	}

	state := &statestore.State{}

	if len(stateFileBytes) == 0 {

		// if file is empty, return the empty state instance
		return &statestore.State{
			InProgress: &sync.Map{},
		}, nil
	}

	// unmarshal
	if err := json.Unmarshal(stateFileBytes, state); err != nil {
		return nil, errors.Wrap(err, "Failed to unmarshal state file")
	}

	return state, nil
}
