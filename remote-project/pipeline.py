

import mlrun
from kfp import dsl
import time
@dsl.pipeline(
    name="hey",
    description="Dede",
)
def pipeline(seconds=5):
    mlrun.run_function(
        "sleep",
        params={"seconds": seconds},
        name="sleep-step",
    )
