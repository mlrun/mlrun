from typing import List

from .callback import Callback, CallbackEnv


class LoggingCallback(Callback):
    def __init__(self, metrics: List[str] = None):
        self._metrics = None

    def __call__(self, env: CallbackEnv) -> None:
        pass

    def