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

package bufferpool

import (
	"testing"

	"github.com/nuclio/logger"
	"github.com/nuclio/loggerus"
	"github.com/stretchr/testify/suite"
)

type sizedBytePoolTestSuite struct {
	suite.Suite
	logger logger.Logger
}

func (suite *sizedBytePoolTestSuite) SetupSuite() {
	var err error
	suite.logger, err = loggerus.NewLoggerusForTests("test")
	suite.Require().NoError(err, "Failed to create logger")

	suite.logger.Info("Setup complete")
}

func (suite *sizedBytePoolTestSuite) TearDownSuite() {
	suite.logger.Info("Tear down complete")
}

func (suite *sizedBytePoolTestSuite) TestByteBufferPoolBlocking() {
	numOfBuffers := 4
	pool := NewSizedBytePool(numOfBuffers, 10)

	var buffers [][]byte

	for i := 0; i < numOfBuffers; i++ {
		buffer := pool.Get()
		suite.Require().NotNil(buffer, "Failed to get buffer from pool")
		buffers = append(buffers, buffer)
	}

	suite.Require().Equal(0, pool.NumPooled(), "Pool should be empty")
	suite.Require().Equal(numOfBuffers, pool.PoolSize(), "Pool size should be %d", numOfBuffers)

	// try to get another buffer in a go routine, should block
	var newBuffer []byte
	notBlocked := make(chan bool)
	go func() {
		newBuffer = pool.Get()
		notBlocked <- true
	}()

	// make sure new buffer is not set yet, and the pool is still full
	suite.Require().Nil(newBuffer, "New buffer should not be set yet")
	suite.Require().Equal(numOfBuffers, pool.PoolSize(), "Pool should be full")
	suite.Require().Equal(0, pool.NumPooled(), "Pool should not have pooled buffers")

	// put the buffers back
	for _, buffer := range buffers {
		pool.Put(buffer)
	}

	// make sure new buffer is set now
	suite.Require().True(<-notBlocked, "Get should not block now")
	suite.Require().NotNil(newBuffer, "New buffer should be set now")

	suite.Require().Equal(numOfBuffers-1, pool.NumPooled(), "Pool should have 3 pooled buffers")

	// put new buffer back
	pool.Put(newBuffer)

	suite.Require().Equal(numOfBuffers, pool.NumPooled(), "Pool should have 4 pooled buffers")
}

func TestLogCollectorTestSuite(t *testing.T) {
	suite.Run(t, new(sizedBytePoolTestSuite))
}
