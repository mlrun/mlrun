from typing import List
import mlrun
from .callback import Callback, CallbackEnv
from ..._ml_common import MLPlan, MLPlanStages


class MLRunCallback(Callback):
    def __init__(self, context: mlrun.MLClientCtx, plans: List[MLPlan]):
        self._context = context
        self._producer = MLLogger

    def __call__(self, env: CallbackEnv) -> None:
        pass
