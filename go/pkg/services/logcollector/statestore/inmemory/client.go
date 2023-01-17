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

	"github.com/mlrun/mlrun/pkg/services/logcollector/statestore"
)

type InMemoryStateStore struct {
	inProgress *sync.Map
}

func NewInMemoryStateStore() *InMemoryStateStore {
	return &InMemoryStateStore{
		inProgress: &sync.Map{},
	}
}

// Initialize initializes the state store
func (i *InMemoryStateStore) Initialize(ctx context.Context) {}

// AddLogItem adds a log item to the state store
func (i *InMemoryStateStore) AddLogItem(ctx context.Context, runId, selector string) error {
	logItem := statestore.LogItem{
		RunUID:        runId,
		LabelSelector: selector,
	}
	i.inProgress.Store(runId, logItem)
	return nil
}

// RemoveLogItem removes a log item from the state store
func (i *InMemoryStateStore) RemoveLogItem(runId string) error {
	i.inProgress.Delete(runId)
	return nil
}

// WriteState writes the state to persistent storage
func (i *InMemoryStateStore) WriteState(state *statestore.State) error {
	return nil
}

// GetItemsInProgress returns the in progress log items
func (i *InMemoryStateStore) GetItemsInProgress() (*sync.Map, error) {
	return i.inProgress, nil
}

// GetState returns the state store state
func (i *InMemoryStateStore) GetState() *statestore.State {
	return &statestore.State{
		InProgress: i.inProgress,
	}
}
