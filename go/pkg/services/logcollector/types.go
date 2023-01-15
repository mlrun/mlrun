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

import "context"

type LogItem struct {
	RunUID        string `json:"runId"`
	LabelSelector string `json:"labelSelector"`
}

type State struct {
	InProgress map[string]LogItem `json:"inProgress"`
}

type StateStore interface {

	// AddLogItem adds a log item to the state store
	AddLogItem(ctx context.Context, runId, selector string) error

	// RemoveLogItem removes a log item from the state store
	RemoveLogItem(runId string) error

	// UpdateState updates the state periodically
	UpdateState(ctx context.Context)

	// WriteState writes the state to persistent storage
	WriteState(state *State) error

	// GetInProgress returns the in progress log items
	GetInProgress() (map[string]LogItem, error)

	// GetState returns the state store state
	GetState() *State
}
