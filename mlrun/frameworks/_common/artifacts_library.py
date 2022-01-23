from abc import ABC, abstractmethod
from typing import Dict, List, Type, Union

import mlrun

from .plan import Plan


class ArtifactsLibrary(ABC):
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
        library_plans = {  # type: Dict[str, Type[Plan]]
            plan_name: plan_class
            for plan_name, plan_class in cls.__dict__.items()
            if isinstance(plan_class, type) and not plan_name.startswith("_")
        }

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
    artifacts_library: Type[ArtifactsLibrary],
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
        if ARTIFACTS_CONTEXT_PARAMETER in context.parameters:
            return artifacts_library.from_dict(
                plans_configuration=context.parameters[ARTIFACTS_CONTEXT_PARAMETER]
            )

    # 3. Return the library's default:
    return artifacts_library.default(**default_kwargs)
