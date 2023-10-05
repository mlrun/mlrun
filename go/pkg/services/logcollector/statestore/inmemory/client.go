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

	"github.com/mlrun/mlrun/pkg/services/logcollector/statestore"
	"github.com/mlrun/mlrun/pkg/services/logcollector/statestore/abstract"

	"github.com/nuclio/logger"
)

type Store struct {
	*abstract.Store
}

func NewInMemoryStore(logger logger.Logger) *Store {
	abstractClient := abstract.NewAbstractClient(logger)
	return &Store{
		Store: abstractClient,
	}
}

// Initialize initializes the state store
func (s *Store) Initialize(ctx context.Context) error {
	return nil
}

// WriteState writes the state to persistent storage
func (s *Store) WriteState(state *statestore.State) error {
	return nil
}

func (s *Store) UpdateLastLogTime(runUID, project string, lastLogTime int64) error {
	return nil
}
