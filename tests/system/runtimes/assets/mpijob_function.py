# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import time

from mpi4py import MPI

import mlrun


@mlrun.handler(outputs=["time", "result"])
def handler(context: mlrun.MLClientCtx):
    # Start the timer:
    run_time = time.time()

    # Get MPI rank:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Log the values (only from root rank (#0) in mpijob):
    if rank == 0:
        time.sleep(1)
        run_time = time.time() - run_time
        return run_time, 1000
