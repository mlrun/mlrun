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

package abstract

import (
	"context"
	"sync"

	"github.com/mlrun/mlrun-go/pkg/common"

	"github.com/mlrun/log-collector/pkg/services/logcollector/statestore"
	"github.com/nuclio/errors"
	"github.com/nuclio/logger"
)

type Store struct {
	Logger           logger.Logger
	AdvancedLogLevel int
	State            *statestore.State
}

func NewAbstractClient(logger logger.Logger, advancedLogLevel int) *Store {
	return &Store{
		Logger:           logger.GetChild("statestore"),
		AdvancedLogLevel: advancedLogLevel,
		State: &statestore.State{
			InProgress: &sync.Map{},
		},
	}
}

// AddLogItem adds a log item to the state store
func (s *Store) AddLogItem(ctx context.Context, runUID, selector, project string) error {
	logItem := statestore.LogItem{
		RunUID:        runUID,
		LabelSelector: selector,
		Project:       project,
	}

	var (
		projectSyncMap *sync.Map
		ok             bool
	)

	// check if project already exists in the state store
	if projectRunUIDsInProgress, projectExists := s.State.InProgress.Load(project); projectExists {
		projectSyncMap, ok = projectRunUIDsInProgress.(*sync.Map)
		if !ok {
			return errors.New("Failed to cast project values in state store to sync.Map")
		}

		// check if run UID already exists in the project
		if existingItem, exists := projectSyncMap.Load(runUID); exists {
			s.Logger.DebugWithCtx(ctx,
				"Item already exists in state file. Overwriting label selector",
				"runUID", runUID,
				"existingItem", existingItem)
		}
	} else {

		// create a new sync map for the project
		projectSyncMap = &sync.Map{}
	}

	// store the log item in the state store
	projectSyncMap.Store(runUID, logItem)
	s.State.InProgress.Store(project, projectSyncMap)
	return nil
}

// RemoveLogItem removes a log item from the state store
func (s *Store) RemoveLogItem(runUID, project string) error {

	var (
		projectSyncMap *sync.Map
		ok             bool
	)

	if projectRunUIDsInProgress, projectExists := s.State.InProgress.Load(project); !projectExists {

		// Project already doesn't exist in state file, nothing to do
		return nil
	} else {
		projectSyncMap, ok = projectRunUIDsInProgress.(*sync.Map)
		if !ok {
			return errors.New("Failed to cast project values in state store to sync.Map")
		}

		// remove run from the project
		projectSyncMap.Delete(runUID)
		s.State.InProgress.Store(project, projectSyncMap)
	}

	// if the project is empty, remove it from the map
	if projectSyncMap != nil && common.SyncMapLength(projectSyncMap) == 0 {
		s.State.InProgress.Delete(project)
	}
	return nil
}

func (s *Store) RemoveProject(project string) error {
	if project == "" {
		return errors.New("Project name is empty")
	}

	// delete is a no-op if the key doesn't exist
	s.State.InProgress.Delete(project)
	return nil
}

// WriteState writes the state to persistent storage
func (s *Store) WriteState(state *statestore.State) error {
	return nil
}

// GetItemsInProgress returns the in progress log items
func (s *Store) GetItemsInProgress() (*sync.Map, error) {
	return s.State.InProgress, nil
}

// GetState returns the state store state
func (s *Store) GetState() *statestore.State {
	return s.State
}
