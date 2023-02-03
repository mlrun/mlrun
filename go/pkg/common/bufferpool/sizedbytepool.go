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

// SizedBytePool is a pool of byte buffers, with a max size
// based on "github.com/oxtoacart/bpool"
type SizedBytePool struct {
	c           chan []byte
	bufferSize  int
	poolSize    int
	currentSize int
}

// NewSizedBytePool creates a new byte buffer pool
func NewSizedBytePool(poolSize, bufferSize int) *SizedBytePool {
	return &SizedBytePool{
		c:          make(chan []byte, poolSize),
		bufferSize: bufferSize,
		poolSize:   poolSize,
	}
}

// Get returns a byte buffer from the pool.
// If the pool is not full, a new buffer is created.
// If the pool is full, the call blocks until a buffer is available.
func (p *SizedBytePool) Get() []byte {
	for {
		select {
		case b := <-p.c:
			return b
		default:
			if p.currentSize < p.poolSize {
				p.currentSize++
				return make([]byte, p.bufferSize)
			}
		}
	}
}

func (p *SizedBytePool) Put(b []byte) {
	if cap(b) < p.bufferSize {
		return
	}
	select {
	case p.c <- b[:p.bufferSize]:
	default:
	}
}

func (p *SizedBytePool) NumPooled() int {
	return len(p.c)
}

func (p *SizedBytePool) PoolSize() int {
	return p.currentSize
}
