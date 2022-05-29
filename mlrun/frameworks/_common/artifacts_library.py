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
    def from_dict(cls, plans_dictionary: Dict[str, dict]) -> List[Plan]:
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

        :param plans_dictionary: The configurations of plans.

        :return: The initialized plans list.

        :raise MLRunInvalidArgumentError: If the configuration was incorrect due to unsupported plan or miss use of
                                          parameters in the plan initializer.
        """
        # Get all of the supported plans in this library:
        library_plans = cls._get_plans()

        # Go through the given configuration an initialize the plans accordingly:
        plans = []  # type: List[Plan]
        for plan_name, plan_parameters in plans_dictionary.items():
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
    def from_list(cls, plans_list: List[str]):
        """
        Initialize a list of plans from a given configuration list. The configuration is expected to be a list of plans
        names to be initialized with their default configuration.

        :param plans_list: The list of plans names to initialize.

        :return: The initialized plans list.

        :raise MLRunInvalidArgumentError: If the configuration was incorrect due to unsupported plan.
        """
        # Get all of the supported plans in this library:
        library_plans = cls._get_plans()

        # Go through the given configuration an initialize the plans accordingly:
        plans = []  # type: List[Plan]
        for plan in plans_list:
            # Initialized plan:
            if isinstance(plan, Plan):
                plans.append(plan)
            # Plan name that needed to be parsed:
            elif isinstance(plan, str):
                # Validate the plan is in the library:
                if plan not in library_plans:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"The given artifact '{plan}' is not supported in this artifacts library. The supported"
                        f"artifacts are: {list(library_plans.keys())}."
                    )
                # Create the plan and collect it:
                plans.append(library_plans[plan]())
            # Unsupported type:
            else:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Expecting a list of artifact plans or plans names from the artifacts library but given: "
                    f"'{type(plan)}'"
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

    @classmethod
    def _get_plans(cls) -> Dict[str, Type[Plan]]:
        """
        Get all of the supported plans in this library.

        :return: The library's plans.
        """
        return {  # type: Dict[str, Type[Plan]]
            plan_name: plan_class
            for plan_name, plan_class in cls.__dict__.items()
            if isinstance(plan_class, type) and not plan_name.startswith("_")
        }


# A constant name for the context parameter to use for passing a plans configuration:
ARTIFACTS_CONTEXT_PARAMETER = "_artifacts"


def get_plans(
    artifacts_library: Type[ArtifactsLibrary],
    artifacts: Union[List[Plan], Dict[str, dict], List[str]] = None,
    context: mlrun.MLClientCtx = None,
    include_default: bool = True,
    **default_kwargs,
) -> List[Plan]:
    """
    Get plans for a run. The plans will be taken from the provided artifacts / configuration via code, from provided
    configuration via MLRun context and if the 'include_default' is True, from the framework artifact library's
    defaults.

    :param artifacts_library: The framework's artifacts library class to get its defaults.
    :param artifacts:         The artifacts parameter passed to the function. Can be passed as a configuration
                              dictionary or an initialized plans list that will simply be returned.
    :param context:           A context to look in if the configuration was passed as a parameter.
    :param include_default:   Whether to include the default in addition to the provided plans. Defaulted to True.
    :param default_kwargs:    Additional key word arguments to pass to the 'default' method of the given artifact
                              library class.

    :return: The plans list.

    :raise MLRunInvalidArgumentError: If the plans were not passed in a list or a dictionary.
    """
    # Setup the plans list:
    parsed_plans = []  # type: List[Plan]

    # Get the user input plans:
    artifacts_from_context = None
    if context is not None:
        artifacts_from_context = context.parameters.get(
            ARTIFACTS_CONTEXT_PARAMETER, None
        )
    for user_input in [artifacts, artifacts_from_context]:
        if user_input is not None:
            if isinstance(user_input, dict):
                parsed_plans += artifacts_library.from_dict(plans_dictionary=user_input)
            elif isinstance(user_input, list):
                parsed_plans += artifacts_library.from_list(plans_list=user_input)
            else:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Artifacts plans are expected to be given in a list or a dictionary, got: '{type(user_input)}'."
                )

    # Get the library's default:
    if include_default:
        parsed_plans += artifacts_library.default(**default_kwargs)

    return parsed_plans
