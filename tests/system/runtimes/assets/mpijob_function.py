import os
import time

import numpy as np
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
