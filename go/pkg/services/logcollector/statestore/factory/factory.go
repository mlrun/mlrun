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

package factory

import (
	"github.com/mlrun/mlrun/pkg/services/logcollector/statestore"
	"github.com/mlrun/mlrun/pkg/services/logcollector/statestore/file"
	"github.com/mlrun/mlrun/pkg/services/logcollector/statestore/inmemory"

	"github.com/nuclio/errors"
)

// CreateStateStore creates a state store of a given kind
func CreateStateStore(
	stateStoreKind statestore.Kind,
	configuration *statestore.Config) (statestore.StateStore, error) {

	switch stateStoreKind {
	case statestore.KindFile:
		return file.NewFileStore(configuration.Logger, configuration.BaseDir, configuration.StateFileUpdateInterval), nil
	case statestore.KindInMemory:
		return inmemory.NewInMemoryStore(configuration.Logger), nil
	default:
		return nil, errors.New("Unknown state store type")
	}

}
