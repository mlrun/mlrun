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
	"encoding/json"
	"os"
	"reflect"
	"sync"
	"time"

	"github.com/mlrun/mlrun/pkg/common"

	"github.com/nuclio/errors"
	"github.com/nuclio/logger"
)

type FileStateStore struct {
	state                   *State
	logger                  logger.Logger
	stateFilePath           string
	lock                    sync.Locker
	stateFileUpdateInterval time.Duration
}

func NewStateFile(logger logger.Logger, filePath string, stateFileUpdateInterval time.Duration) *FileStateStore {
	return &FileStateStore{
		state: &State{
			InProgress: make(map[string]LogItem),
		},
		logger:                  logger.GetChild("filestatestore"),
		stateFilePath:           filePath,
		stateFileUpdateInterval: stateFileUpdateInterval,
		lock:                    &sync.Mutex{},
	}
}

func (f *FileStateStore) AddLogItem(ctx context.Context, runId, selector string) error {
	logItem := LogItem{
		RunId:         runId,
		LabelSelector: selector,
	}

	if existingItem, exists := f.state.InProgress[runId]; exists {
		f.logger.DebugWithCtx(ctx,
			"Item already exists in state file. Overwriting label selector",
			"runId", runId,
			"existingItem", existingItem)
	}

	f.lock.Lock()
	defer f.lock.Unlock()
	f.state.InProgress[logItem.RunId] = logItem
	return nil
}

func (f *FileStateStore) RemoveLogItem(runId string) error {
	f.lock.Lock()
	defer f.lock.Unlock()
	delete(f.state.InProgress, runId)
	return nil
}

func (f *FileStateStore) UpdateState(ctx context.Context) {

	prevState := *f.state

	for {

		// get state
		currentState := *f.state

		// if state changed, write it to file
		if !reflect.DeepEqual(currentState, prevState) {

			prevState = currentState

			// write state file
			if err := f.writeStateToFile(ctx, &currentState); err != nil {
				f.logger.ErrorWithCtx(ctx, "Failed to write state file", "err", err)
				return
			}
		}

		time.Sleep(f.stateFileUpdateInterval)
	}
}

// WriteState writes the state to file, used mainly for testing
func (f *FileStateStore) WriteState(state *State) error {
	return f.writeStateToFile(context.Background(), state)
}

func (f *FileStateStore) GetInProgress() (map[string]LogItem, error) {
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

func (f *FileStateStore) GetState() *State {
	return f.state
}

// writeStateToFile writes the state to file
func (f *FileStateStore) writeStateToFile(ctx context.Context, state *State) error {

	// marshal state file
	encodedState, err := json.Marshal(state)
	if err != nil {
		return errors.Wrap(err, "Failed to encode state file")
	}

	// get lock, unlock later
	f.lock.Lock()
	defer f.lock.Unlock()

	// write to file
	return common.WriteToFile(ctx, f.logger, f.stateFilePath, f.lock, encodedState, false)
}

// readStateFile reads the state from the file
func (f *FileStateStore) readStateFile() (*State, error) {

	// get lock
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
