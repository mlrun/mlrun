from abc import ABC, abstractmethod
from typing import Collection

import mlrun


class MetricsLibrary(ABC):
    """
    Static class for parsing user set metrics of a framework.
    """

    # A constant name for the context parameter to use for passing a plans configuration:
    CONTEXT_PARAMETER = "_metrics"

    @classmethod
    def get_metrics(cls, context: mlrun.MLClientCtx) -> Collection:
        """
        Get metrics for a run. The metrics will be taken from the provided configuration via a MLRun context.

        :param context: A context to look in if the configuration was passed as a parameter.

        :return: The parsed collection of metrics in the context.
        """
        metrics_from_context = context.parameters.get(cls.CONTEXT_PARAMETER, None)
        return cls._parse(metrics=metrics_from_context) if metrics_from_context is not None else None

    @staticmethod
    @abstractmethod
    def _parse(metrics) -> Collection:
        """
        Parse the given metrics by the possible rules of the framework implementing.

        :param metrics: A collection of metrics to parse.

        :return: The parsed metrics to use in training / evaluation.
        """
        pass
