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

package inmemory

import (
	"context"

	"github.com/mlrun/log-collector/pkg/services/logcollector/statestore"
	"github.com/mlrun/log-collector/pkg/services/logcollector/statestore/abstract"
)

type Store struct {
	*abstract.Store
}

func NewInMemoryStore(configuration *statestore.Config) *Store {
	abstractClient := abstract.NewAbstractClient(configuration.Logger, configuration.AdvancedLogLevel)
	return &Store{
		Store: abstractClient,
	}
}

// Initialize initializes the state store
func (s *Store) Initialize(ctx context.Context) error {
	return nil
}

func (s *Store) AddLogItem(ctx context.Context, runUID, selector, project string) error {
	if s.Store.AdvancedLogLevel >= 1 {
		s.Logger.DebugWithCtx(ctx,
			"Adding item to in memory state",
			"runUID", runUID,
			"selector", selector,
			"project", project)
	}
	return s.Store.AddLogItem(ctx, runUID, selector, project)
}

func (s *Store) RemoveLogItem(ctx context.Context, runUID, project string) error {
	if s.Store.AdvancedLogLevel >= 1 {
		s.Logger.DebugWithCtx(ctx,
			"Removing item from in memory state",
			"runUID", runUID,
			"project", project)
	}
	return s.Store.RemoveLogItem(runUID, project)
}

// WriteState writes the state to persistent storage
func (s *Store) WriteState(state *statestore.State) error {
	return nil
}

func (s *Store) UpdateLastLogTime(runUID, project string, lastLogTime int64) error {
	return nil
}
