# Copyright 2023 Iguazio
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


def handler():
    """
    A handler to check the `MPIJobRuntime` as a system test in `TestMpiJobRuntime.test_mpijob_run`. It will:

    * Test the OpenMPI 4 setup by getting the rank and reducing from all ranks to the root rank.
    * Test the run state changes to completed only after the last worker finished.
    * Test the `mlrun.package` mechanism for logging returning values that were returned only from the logging worker -
      which is defaulted to rank 0.

    If the handler ran with 4 workers (4 replicas) the returning values should be:
    40 (10+10+10+10), 1000 ((0 + 1) * 1000).
    """
    # Get MPI rank:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # To check all workers were initialized and the MPI protocol is set correctly, we collect the value 10 from all
    # replicas to the root where it will be logged when returned:
    result = 10
    result = comm.reduce(result, op=MPI.SUM, root=0)

    # Make rank 0 sleep for some time to make sure all the other ranks will finish before it to test the run state
    # completion race case:
    if rank == 0:
        time.sleep(30)

    # Return the accumulated value and a per rank value. Notice: only values returned from root rank (#0) should be
    # logged (see `mlrun.execution.MLClientCtx.is_logging_worker`):
    return result, 1000 * (rank + 1)
