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

type Pool interface {

	// Get returns a buffer from the pool.
	Get() []byte

	// Put returns a buffer to the pool.
	Put([]byte)

	// NumPooled returns the number of buffers currently in the pool.
	NumPooled() int

	// PoolSize returns the maximum number of buffers in the pool.
	PoolSize() int
}
