from typing import List
import mlrun
from .callback import CallbackEnv
from .logging_callback import LoggingCallback
from ..._ml_common import MLPlan, MLPlanStages, MLProducer


class MLRunCallback(LoggingCallback):
    def __init__(self, context: mlrun.MLClientCtx, plans: List[MLPlan]):
        self._context = context
        self._producer = MLProducer(context=context, plans=plans)

    def __call__(self, env: CallbackEnv) -> None:
        pass
