//go:build test_unit

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
	"os"
	"path"
	"testing"
	"time"

	"github.com/mlrun/mlrun/pkg/common"
	"github.com/mlrun/mlrun/pkg/services/logcollector/statestore"

	"github.com/nuclio/logger"
	"github.com/nuclio/loggerus"
	"github.com/stretchr/testify/suite"
)

type FileStateStoreTestSuite struct {
	suite.Suite
	logger     logger.Logger
	ctx        context.Context
	baseDir    string
	stateStore *Store
}

func (suite *FileStateStoreTestSuite) SetupTest() {
	var err error
	suite.logger, err = loggerus.NewLoggerusForTests("test")
	suite.Require().NoError(err, "Failed to create logger")

	suite.ctx = context.Background()

	// create base dir
	suite.baseDir = path.Join(os.TempDir(), "/log_collector_test")
	err = os.MkdirAll(suite.baseDir, 0777)
	suite.Require().NoError(err, "Failed to create base dir")

	// create state store
	suite.stateStore = NewFileStore(suite.logger, suite.baseDir, 2*time.Second)
	suite.stateStore.Initialize(suite.ctx)

	suite.logger.InfoWith("Setup complete")
}

func (suite *FileStateStoreTestSuite) TearDownTest() {
	suite.logger.DebugWith("Tearing down test")

	// delete base dir and created files
	err := os.RemoveAll(suite.baseDir)
	suite.Require().NoError(err, "Failed to delete base dir")

	suite.logger.InfoWith("Tear down complete")

}

func (suite *FileStateStoreTestSuite) TestReadWriteStateFile() {

	// read state file
	logItemsInProgress, err := suite.stateStore.GetItemsInProgress()
	suite.Require().NoError(err, "Failed to read state file")

	// verify no items in progress
	suite.Require().Equal(0, common.SyncMapLength(logItemsInProgress))

	// add a log item to the state file
	runId := "abc123"
	item := statestore.LogItem{
		RunUID:        runId,
		LabelSelector: "app=test",
	}

	logItemsInProgress.Store(runId, item)

	// write state file
	err = suite.stateStore.WriteState(&statestore.State{
		InProgress: logItemsInProgress,
	})
	suite.Require().NoError(err, "Failed to write state file")

	// read state file again
	logItemsInProgress, err = suite.stateStore.GetItemsInProgress()
	suite.Require().NoError(err, "Failed to read state file")

	// verify item is in progress
	suite.Require().Equal(1, common.SyncMapLength(logItemsInProgress))
	storedItem, ok := logItemsInProgress.Load(runId)
	suite.Require().True(ok)
	suite.Require().Equal(item, storedItem.(statestore.LogItem))
}

func (suite *FileStateStoreTestSuite) TestAddRemoveItemFromInProgress() {
	runId := "some-run-id"
	labelSelector := "app=test"

	err := suite.stateStore.AddLogItem(suite.ctx, runId, labelSelector)
	suite.Require().NoError(err, "Failed to add item to in progress")

	// write state to file
	err = suite.stateStore.WriteState(suite.stateStore.GetState())
	suite.Require().NoError(err, "Failed to write state file")

	// read state file
	itemsInProgress, err := suite.stateStore.GetItemsInProgress()
	suite.Require().NoError(err, "Failed to read state file")

	// verify item is in progress
	suite.Require().Equal(1, common.SyncMapLength(itemsInProgress))
	storedItem, ok := itemsInProgress.Load(runId)
	suite.Require().True(ok)
	suite.Require().Equal(runId, storedItem.(statestore.LogItem).RunUID)
	suite.Require().Equal(labelSelector, storedItem.(statestore.LogItem).LabelSelector)

	// remove item from in progress
	err = suite.stateStore.RemoveLogItem(runId)
	suite.Require().NoError(err, "Failed to remove item from in progress")

	// write state to file again
	err = suite.stateStore.WriteState(suite.stateStore.GetState())
	suite.Require().NoError(err, "Failed to write state file")

	// read state file again
	itemsInProgress, err = suite.stateStore.GetItemsInProgress()
	suite.Require().NoError(err, "Failed to read state file")

	// verify item is not in progress
	suite.Require().Equal(0, common.SyncMapLength(itemsInProgress))
}

func TestFileStateStoreTestSuite(t *testing.T) {
	suite.Run(t, new(FileStateStoreTestSuite))
}
