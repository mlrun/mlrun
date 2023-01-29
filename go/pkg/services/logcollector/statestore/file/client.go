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
	"sync"
	"time"

	"github.com/mlrun/mlrun/pkg/common"
	"github.com/mlrun/mlrun/pkg/services/logcollector/statestore"

	"github.com/nuclio/errors"
	"github.com/nuclio/logger"
)

type Store struct {
	state                   *statestore.State
	logger                  logger.Logger
	stateFilePath           string
	stateFileUpdateInterval time.Duration
	lock                    sync.Locker
}

func NewFileStore(logger logger.Logger, baseDirPath string, stateFileUpdateInterval time.Duration) *Store {
	return &Store{
		state: &statestore.State{
			InProgress: &sync.Map{},
		},
		logger: logger.GetChild("filestatestore"),
		// setting _metadata with "_" as a sub directory, so it won't conflict with projects directories
		stateFilePath:           path.Join(baseDirPath, "_metadata", "state.json"),
		stateFileUpdateInterval: stateFileUpdateInterval,
		lock:                    &sync.Mutex{},
	}
}

// Initialize initializes the file state store
func (s *Store) Initialize(ctx context.Context) {

	// spawn a goroutine that will update the state file periodically
	go s.stateFileUpdateLoop(ctx)
}

// AddLogItem adds a log item to the state store
func (s *Store) AddLogItem(ctx context.Context, runUID, selector string) error {
	logItem := statestore.LogItem{
		RunUID:        runUID,
		LabelSelector: selector,
	}

	if existingItem, exists := s.state.InProgress.Load(runUID); exists {
		s.logger.DebugWithCtx(ctx,
			"Item already exists in state file. Overwriting label selector",
			"runUID", runUID,
			"existingItem", existingItem)
	}

	s.state.InProgress.Store(logItem.RunUID, logItem)
	return nil
}

// RemoveLogItem removes a log item from the state store
func (s *Store) RemoveLogItem(runUID string) error {
	s.state.InProgress.Delete(runUID)
	return nil
}

// WriteState writes the state to file, used mainly for testing
func (s *Store) WriteState(state *statestore.State) error {
	return s.writeStateToFile(state)
}

// GetItemsInProgress returns the in progress log items
func (s *Store) GetItemsInProgress() (*sync.Map, error) {
	state, err := s.readStateFile()
	if err != nil {
		return nil, errors.Wrap(err, "Failed to read state file")
	}

	// set the state in the file state store
	s.lock.Lock()
	defer s.lock.Unlock()
	s.state = state

	return s.state.InProgress, nil
}

// GetState returns the state store state
func (s *Store) GetState() *statestore.State {
	s.lock.Lock()
	defer s.lock.Unlock()
	return s.state
}

// stateFileUpdateLoop updates the state file periodically
func (s *Store) stateFileUpdateLoop(ctx context.Context) {

	// create ticker
	ticker := time.NewTicker(s.stateFileUpdateInterval)
	defer ticker.Stop()

	// count the errors so we won't spam the logs
	errCount := 0

	for range ticker.C {

		// get state
		state := s.GetState()

		// write state to file
		if err := s.writeStateToFile(state); err != nil {
			if errCount%5 == 0 {
				errCount = 0
				s.logger.WarnWithCtx(ctx,
					"Failed to write state file",
					"err", err.Error())
			}
			errCount++
		}
	}
}

// writeStateToFile writes the state to file
func (s *Store) writeStateToFile(state *statestore.State) error {

	// marshal state file
	encodedState, err := json.Marshal(state)
	if err != nil {
		return errors.Wrap(err, "Failed to encode state file")
	}

	// get lock, unlock later
	s.lock.Lock()
	defer s.lock.Unlock()

	// write to file
	return common.WriteToFile(s.stateFilePath, encodedState, false)
}

// readStateFile reads the state from the file
func (s *Store) readStateFile() (*statestore.State, error) {

	// get lock so file won't be updated while reading
	s.lock.Lock()
	defer s.lock.Unlock()

	// read file
	if err := common.EnsureFileExists(s.stateFilePath); err != nil {
		return nil, errors.Wrap(err, "Failed to ensure state file exists")
	}
	stateFileBytes, err := os.ReadFile(s.stateFilePath)
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
