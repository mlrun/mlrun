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

package inmemory

import (
	"context"
	"sync"

	"github.com/mlrun/mlrun/pkg/common"
	"github.com/mlrun/mlrun/pkg/services/logcollector/statestore"

	"github.com/nuclio/errors"
)

type Store struct {
	state *statestore.State
	lock  sync.Locker
}

func NewInMemoryStore() *Store {
	return &Store{
		state: &statestore.State{
			InProgress: map[string]*sync.Map{},
		},
		lock: &sync.Mutex{},
	}
}

// Initialize initializes the state store
func (s *Store) Initialize(ctx context.Context) error {
	return nil
}

// AddLogItem adds a log item to the state store
func (s *Store) AddLogItem(ctx context.Context, runUID, selector, project string) error {
	logItem := statestore.LogItem{
		RunUID:        runUID,
		LabelSelector: selector,
		Project:       project,
	}
	s.lock.Lock()
	defer s.lock.Unlock()

	if _, projectExists := s.state.InProgress[project]; !projectExists {
		s.state.InProgress[project] = &sync.Map{}
	}

	s.state.InProgress[project].Store(runUID, logItem)
	return nil
}

// RemoveLogItem removes a log item from the state store
func (s *Store) RemoveLogItem(runUID, project string) error {
	s.lock.Lock()
	defer s.lock.Unlock()

	if _, exists := s.state.InProgress[project]; !exists {
		return errors.New("Project not found")
	}

	s.state.InProgress[project].Delete(runUID)

	// if the project is empty, remove it from the map
	if common.SyncMapLength(s.state.InProgress[project]) == 0 {
		delete(s.state.InProgress, project)
	}

	return nil
}

// WriteState writes the state to persistent storage
func (s *Store) WriteState(state *statestore.State) error {
	return nil
}

// GetItemsInProgress returns the in progress log items
func (s *Store) GetItemsInProgress() (map[string]*sync.Map, error) {
	return s.state.InProgress, nil
}

// GetState returns the state store state
func (s *Store) GetState() *statestore.State {
	return s.state
}
