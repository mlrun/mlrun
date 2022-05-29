from abc import ABC, abstractmethod
from typing import Dict

import mlrun
from mlrun.artifacts import Artifact
from mlrun.utils.helpers import is_ipython


class Plan(ABC):
    """
    An abstract class for describing a plan. A plan is used to produce artifact manually or in a given time of a
    function according to its configuration.
    """

    def __init__(self):
        """
        Initialize a new plan.
        """
        self._artifacts = {}  # type: Dict[str, Artifact]

    @property
    def artifacts(self) -> Dict[str, Artifact]:
        """
        Get the plan's produced artifacts.

        :return: The plan's artifacts.
        """
        return self._artifacts

    def is_reproducible(self, *args, **kwargs) -> bool:
        """
        Check whether or not the plan should be used to produce multiple times or only once. Defaulted to return False.

        :return: True if the plan is reproducible and False otherwise.
        """
        return False

    @abstractmethod
    def is_ready(self, *args, **kwargs) -> bool:
        """
        Check whether or not the plan is fit for production in the current time this method is called.

        :return: True if the plan is producible and False otherwise.
        """
        pass

    @abstractmethod
    def produce(self, *args, **kwargs) -> Dict[str, Artifact]:
        """
        Produce the artifact according to this plan.

        :return: The produced artifacts.
        """
        pass

    def log(self, context: mlrun.MLClientCtx):
        """
        Log the artifacts in this plan to the given context.

        :param context: A MLRun context to log with.
        """
        for artifact_name, artifact_object in self._artifacts.items():
            context.log_artifact(artifact_object)

    def display(self):
        """
        Display the plan's artifact. If artifacts were not produced nothing will be printed.
        """
        # Validate at least one artifact was produced:
        if not self._artifacts:
            return

        # Call the correct display method according to the kernel:
        if is_ipython:
            self._gui_display()
        else:
            self._cli_display()

    @abstractmethod
    def _cli_display(self):
        """
        How the plan's products would be presented on a command line kernel.
        """
        pass

    @abstractmethod
    def _gui_display(self):
        """
        How the plan's products would be presented on a graphic IPython kernel (like a Jupyter notebook).
        """
        pass

    def _repr_pretty_(self, p, cycle: bool):
        """
        A pretty representation of the plan. Will be called by the IPython kernel. This method will call the plan's
        display method.

        :param p:     A RepresentationPrinter instance.
        :param cycle: If a cycle is detected to prevent infinite loop.
        """
        self.display()
