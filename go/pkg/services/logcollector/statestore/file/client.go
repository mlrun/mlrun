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
	"github.com/mlrun/mlrun/pkg/services/logcollector/statestore/abstract"

	"github.com/nuclio/errors"
	"github.com/nuclio/logger"
)

type Store struct {
	*abstract.Store
	stateFilePath           string
	stateFileUpdateInterval time.Duration
	fileLock                sync.Locker
	stateLock               sync.Locker
}

func NewFileStore(logger logger.Logger, baseDirPath string, stateFileUpdateInterval time.Duration) *Store {
	abstractClient := abstract.NewAbstractClient(logger)
	return &Store{
		Store: abstractClient,
		// setting _metadata with "_" as a subdirectory, so it won't conflict with projects directories
		stateFilePath:           path.Join(baseDirPath, "_metadata", "state.json"),
		stateFileUpdateInterval: stateFileUpdateInterval,
		fileLock:                &sync.Mutex{},
		stateLock:               &sync.Mutex{},
	}
}

// Initialize initializes the file state store
func (s *Store) Initialize(ctx context.Context) error {
	var err error

	// load state from file before starting the update loop, as thr file is our source of truth
	s.fileLock.Lock()
	s.State, err = s.readStateFile()
	if err != nil {
		s.fileLock.Unlock()
		return errors.Wrap(err, "Failed to read state file")
	}

	// we do not defer unlock because we need the lock for writing the state to file
	s.fileLock.Unlock()

	// write state to file to make sure it exists
	if err := s.writeStateToFile(s.State); err != nil {
		return errors.Wrap(err, "Failed to write state file")
	}

	// spawn a goroutine that will update the state file periodically
	go s.stateFileUpdateLoop(ctx)

	s.Logger.DebugWithCtx(ctx, "Successfully initialized file state store")

	return nil
}

// WriteState writes the state to file, used mainly for testing
func (s *Store) WriteState(state *statestore.State) error {
	return s.writeStateToFile(state)
}

// GetState returns the state store state
func (s *Store) GetState() *statestore.State {
	s.stateLock.Lock()
	defer func() {
		s.stateLock.Unlock()
	}()
	return s.State
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
				s.Logger.WarnWithCtx(ctx,
					"Failed to write state file",
					"err", common.GetErrorStack(err, common.DefaultErrorStackDepth),
				)
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
	s.fileLock.Lock()
	defer func() {
		s.fileLock.Unlock()
	}()

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
