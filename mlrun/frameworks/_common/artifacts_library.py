import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, Union

import mlrun
import mlrun.errors
from mlrun.artifacts import Artifact


class Plan(ABC):
    """
    An abstract class for describing a plan. A plan is used to produce artifact in a given time of a function according
    to its configuration.
    """

    def __init__(self, auto_produce: bool = True, **produce_arguments):
        """
        Initialize a new plan. The plan will be automatically produced if all of the required arguments to the produce
        method are given.

        :param auto_produce:      Whether to automatically produce the artifact if all of the required arguments are
                                  given. Defaulted to True.
        :param produce_arguments: The provided arguments to the produce method in kwargs style.
        """
        # Set the artifacts dictionary:
        self._artifacts = {}  # type: Dict[str, Artifact]

        # Check if the plan should be produced, if so call produce:
        if auto_produce and self._is_producible(given_parameters=produce_arguments):
            self.produce(**produce_arguments)

    @property
    def artifacts(self) -> Dict[str, Artifact]:
        """
        Get the plan's produced artifacts.

        :return: The plan's artifacts.
        """
        return self._artifacts

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
        Display the plan's artifact. This method will be called by the IPython kernel to showcase the plan. If artifacts
        were produced this is the method to draw and print them.
        """
        print(repr(self))

    def _is_producible(self, given_parameters: Dict[str, Any]) -> bool:
        """
        Check if the plan can call its 'produce' method with the given parameters. If all of the required parameters are
        given, True is returned and Otherwise, False.

        :param given_parameters: The given parameters to check.

        :return: True if the plan is producible and False if not.

        :raise MLRunInvalidArgumentError: If some of the required parameters were provided but not all of them in order
                                          to make sure a proper usage of the initialization method.
        """
        # Get this plan's produce method required parameters:
        required_parameters = [
            parameter_name
            for parameter_name, parameter_object in inspect.signature(
                self.produce
            ).parameters.items()
            if parameter_object.default == parameter_object.empty
            and parameter_object.name not in ["args", "kwargs"]
        ]

        # Parse the given parameters into a list of the actual given (not None) parameters:
        given_parameters = [
            parameter_name
            for parameter_name, parameter_value in given_parameters.items()
            if parameter_value is not None
        ]

        # Validate that if some of the required parameters are passed, all of them are:
        missing_parameters = [
            required_parameter
            for required_parameter in required_parameters
            if required_parameter not in given_parameters
        ]
        if 0 < len(missing_parameters) < len(required_parameters):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Artifact cannot be produced. Some of the required arguments are missing: {missing_parameters}."
            )

        # Return True only if all the required parameters are given:
        return len(missing_parameters) == 0

    def _repr_pretty_(self, p, cycle: bool):
        """
        A pretty representation of the plan. Will be called by the IPython kernel. This method will call the plan's
        display method.

        :param p:     A RepresentationPrinter instance.
        :param cycle: If a cycle is detected to prevent infinite loop.
        """
        self.display()


class ArtifactLibrary(ABC):
    """
    An abstract class for an artifacts library. Each framework should have an artifacts library for knowing what
    artifacts can be produced and their configurations. The default method must be implemented for when the user do not
    pass any plans.

    To add a plan to the library, simply write its name in the library class as a class variable pointing to the plan's
    class 'init_artifact' class method:

    some_artifact = SomeArtifactPlan
    """

    @classmethod
    def from_dict(cls, plans_configuration: Dict[str, dict]) -> List[Plan]:
        """
        Initialize a list of plans from a given configuration dictionary. The configuration is expected to be a
        dictionary of plans and their initialization parameters in the following format:
        {
            PLAN_NAME: {
                PARAMETER_NAME: PARAMETER_VALUE,
                ...
            },
            ...
        }

        :param plans_configuration: The configurations of plans.

        :return: The initialized plans list.

        :raise MLRunInvalidArgumentError: If the configuration was incorrect due to unsupported plan or miss use of
                                          parameters in the plan initializer.
        """
        # Get all of the supported plans in this library:
        library_plans = {
            plan_name: plan_class
            for plan_name, plan_class in cls.__dict__.items()
            if isinstance(plan_class, type) and not plan_name.startswith("_")
        }  # type: Dict[str, Type[Plan]]

        # Go through the given configuration an initialize the plans accordingly:
        plans = []  # type: List[Plan]
        for plan_name, plan_parameters in plans_configuration.items():
            # Validate the plan is in the library:
            if plan_name not in library_plans:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"The given artifact '{plan_name}' is not supported in this artifacts library. The supported"
                    f"artifacts are: {list(library_plans.keys())}."
                )
            # Try to create the plan with the given parameters:
            try:
                plans.append(library_plans[plan_name](**plan_parameters))
            except TypeError as error:
                # A TypeError was raised, that means there was a miss use of parameters in the plan's '__init__' method:
                raise mlrun.MLRunInvalidArgumentError(
                    f"The following artifact: '{plan_name}' cannot be parsed due to miss use of parameters: {error}"
                )

        return plans

    @classmethod
    @abstractmethod
    def default(cls, **kwargs) -> List[Plan]:
        """
        Get the default artifacts plans list of this framework's library.

        :return: The default artifacts plans list.
        """
        pass


# A constant name for the context parameter to use for passing a plans configuration:
ARTIFACTS_CONTEXT_PARAMETER = "artifacts"


def get_plans(
    artifacts_library: Type[ArtifactLibrary],
    artifacts: Union[List[Plan], Dict[str, dict]] = None,
    context: mlrun.MLClientCtx = None,
    **default_kwargs,
):
    """
    Get plans for a run by the following priority:

    1. Provided artifacts / configuration via code.
    2. Provided configuration via MLRun context.
    3. The framework artifact library's defaults.

    :param artifacts_library: The framework's artifacts library class to get its defaults.
    :param artifacts:         The artifacts parameter passed to the function. Can be passed as a configuration
                              dictionary or an initialized plans list that will simply be returned.
    :param context:           A context to look in if the configuration was passed as a parameter.
    :param default_kwargs:    Additional key word arguments to pass to the 'default' method of the given artifact
                              library class.

    :return: The plans list by the priority mentioned above.
    """
    # 1. Try from given artifacts:
    if artifacts is not None:
        if isinstance(artifacts, dict):
            return artifacts_library.from_dict(plans_configuration=artifacts)
        return artifacts

    # 2. Try from passed context:
    if context is not None:
        context_parameters = context.parameters
        if ARTIFACTS_CONTEXT_PARAMETER in context_parameters:
            return artifacts_library.from_dict(
                plans_configuration=context.parameters[ARTIFACTS_CONTEXT_PARAMETER]
            )

    # 3. Return the library's default:
    return artifacts_library.default(**default_kwargs)
