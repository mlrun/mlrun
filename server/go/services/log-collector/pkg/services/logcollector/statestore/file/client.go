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

package file

import (
	"context"
	"encoding/json"
	"os"
	"path"
	"sync"
	"time"

	"github.com/mlrun/mlrun-go/pkg/common"

	"github.com/mlrun/log-collector/pkg/services/logcollector/statestore"
	"github.com/mlrun/log-collector/pkg/services/logcollector/statestore/abstract"
	"github.com/nuclio/errors"
)

type Store struct {
	*abstract.Store
	stateFilePath           string
	stateFileUpdateInterval time.Duration
	fileLock                sync.Locker
	stateLock               sync.Locker
}

func NewFileStore(configuration *statestore.Config) *Store {
	abstractClient := abstract.NewAbstractClient(configuration.Logger, configuration.AdvancedLogLevel)
	return &Store{
		Store: abstractClient,
		// setting _metadata with "_" as a subdirectory, so it won't conflict with projects directories
		stateFilePath:           path.Join(configuration.BaseDir, "_metadata", "state.json"),
		stateFileUpdateInterval: configuration.StateFileUpdateInterval,
		fileLock:                &sync.Mutex{},
		stateLock:               &sync.Mutex{},
	}
}

// Initialize initializes the file state store
func (s *Store) Initialize(ctx context.Context) error {
	var err error

	// lock the file for the duration of the initialization
	s.fileLock.Lock()
	defer s.fileLock.Unlock()

	// load state from file before starting the update loop, as the file is our source of truth
	s.State, err = s.readStateFile()
	if err != nil {
		return errors.Wrap(err, "Failed to read state file")
	}

	// write state to file to make sure it exists
	if err := s.writeStateToFile(s.State); err != nil {
		return errors.Wrap(err, "Failed to write state file")
	}

	// spawn a goroutine that will update the state file periodically
	go s.stateFileUpdateLoop(ctx)

	s.Logger.DebugWithCtx(ctx, "Successfully initialized file state store")

	return nil
}

func (s *Store) AddLogItem(ctx context.Context, runUID, selector, project string) error {
	if s.Store.AdvancedLogLevel >= 1 {
		s.Logger.DebugWithCtx(ctx,
			"Adding item to state file",
			"runUID", runUID,
			"selector", selector,
			"project", project)
	}
	return s.Store.AddLogItem(ctx, runUID, selector, project)
}

func (s *Store) RemoveLogItem(ctx context.Context, runUID, project string) error {
	if s.Store.AdvancedLogLevel >= 1 {
		s.Logger.DebugWithCtx(ctx,
			"Removing item from state file",
			"runUID", runUID,
			"project", project)
	}
	return s.Store.RemoveLogItem(runUID, project)
}

// WriteState writes the state to file, used mainly for testing
func (s *Store) WriteState(state *statestore.State) error {
	return s.writeStateToFile(state)
}

// GetState returns the state store state
func (s *Store) GetState() *statestore.State {
	s.stateLock.Lock()
	defer s.stateLock.Unlock()
	return s.State
}

func (s *Store) UpdateLastLogTime(runUID, project string, lastLogTime int64) error {

	// get the state file of the run
	if projectRunUIDsInProgress, projectExists := s.State.InProgress.Load(project); projectExists {
		projectSyncMap, ok := projectRunUIDsInProgress.(*sync.Map)
		if !ok {
			return errors.New("Failed to cast project values in state store to sync.Map")
		}

		// check if run UID already exists in the project
		if runLogItem, runLogItemExists := projectSyncMap.Load(runUID); runLogItemExists {
			runLogItem, ok := runLogItem.(statestore.LogItem)
			if !ok {
				return errors.New("Failed to cast run log item to statestore.LogItem")
			}
			runLogItem.LastLogTime = lastLogTime
			projectSyncMap.Store(runUID, runLogItem)

			// update the state file
			s.State.InProgress.Store(project, projectSyncMap)
		} else {
			return errors.New("Failed to find run UID in state store")
		}
	} else {
		return errors.New("Failed to find project in state store")
	}

	return nil
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
		s.fileLock.Lock()
		if err := s.writeStateToFile(state); err != nil {
			if errCount%5 == 0 {
				errCount = 0
				s.Logger.WarnWithCtx(ctx,
					"Failed to write state file",
					"err", common.GetErrorStack(err, common.DefaultErrorStackDepth),
				)
			}
			errCount++
		}
		s.fileLock.Unlock()
	}
}

// writeStateToFile writes the state to file
func (s *Store) writeStateToFile(state *statestore.State) error {

	// marshal state file
	encodedState, err := json.Marshal(state)
	if err != nil {
		return errors.Wrap(err, "Failed to encode state file")
	}

	// write to file
	return common.WriteToFile(s.stateFilePath, encodedState, false)
}

// readStateFile reads the state from the file
func (s *Store) readStateFile() (*statestore.State, error) {

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
