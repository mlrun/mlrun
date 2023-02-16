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

package statestore

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/nuclio/errors"
	"github.com/nuclio/logger"
)

type Kind string

const (
	KindFile     Kind = "file"
	KindInMemory Kind = "inMemory"
)

type LogItem struct {
	RunUID        string `json:"runUID"`
	LabelSelector string `json:"labelSelector"`
	Project       string `json:"project"`
}

// MarshalledState is a helper struct for marshalling the state
type marshalledState struct {
	InProgress map[string]LogItem `json:"inProgress"`
}

type State struct {
	InProgress *sync.Map `json:"inProgress"`
}

// UnmarshalJSON is a custom unmarshaler for the state
func (s *State) UnmarshalJSON(data []byte) error {
	tempState := marshalledState{
		InProgress: map[string]LogItem{},
	}

	s.InProgress = &sync.Map{}

	if err := json.Unmarshal(data, &tempState); err != nil {
		return errors.Wrap(err, "Failed to unmarshal data")
	}
	for key, value := range tempState.InProgress {
		s.InProgress.Store(key, value)
	}
	return nil
}

// MarshalJSON is a custom marshaler for the state
func (s *State) MarshalJSON() ([]byte, error) {

	tempState := marshalledState{
		InProgress: map[string]LogItem{},
	}

	s.InProgress.Range(func(key, value interface{}) bool {
		keyString, ok := key.(string)
		if !ok {
			return false
		}
		valueLogItem, ok := value.(LogItem)
		if !ok {
			return false
		}
		tempState.InProgress[keyString] = valueLogItem
		return true
	})
	return json.Marshal(tempState)
}

type Config struct {
	Logger                  logger.Logger
	StateFileUpdateInterval time.Duration
	BaseDir                 string
}

type StateStore interface {

	// Initialize initializes the state store
	Initialize(ctx context.Context) error

	// AddLogItem adds a log item to the state store
	AddLogItem(ctx context.Context, runUID, selector, project string) error

	// RemoveLogItem removes a log item from the state store
	RemoveLogItem(runUID, project string) error

	// WriteState writes the state to persistent storage
	WriteState(state *State) error

	// GetItemsInProgress returns the in progress log items
	GetItemsInProgress() (*sync.Map, error)

	// GetState returns the state store state
	GetState() *State
}

func GenerateKey(runUID, project string) string {
	return fmt.Sprintf("%s/%s", runUID, project)
}
