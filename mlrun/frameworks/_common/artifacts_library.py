# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from abc import ABC, abstractmethod
from typing import Optional, Union

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

    # A constant name for the context parameter to use for passing a plans configuration:
    CONTEXT_PARAMETER = "_artifacts"

    # TODO: Finish support for custom plans.
    @classmethod
    def get_plans(
        cls,
        artifacts: Optional[Union[list[Plan], dict[str, dict], list[str]]] = None,
        context: mlrun.MLClientCtx = None,
        include_default: bool = True,
        # custom_plans: dict = None, :param custom_plans: Custom user plans objects to initialize from.
        **default_kwargs,
    ) -> list[Plan]:
        """
        Get plans for a run. The plans will be taken from the provided artifacts / configuration via code, from provided
        configuration via MLRun context and if the 'include_default' is True, from the framework artifact library's
        defaults.

        :param artifacts:         The artifacts parameter passed to the function. Can be passed as a configuration
                                  dictionary or an initialized plans list that will simply be returned.
        :param context:           A context to look in if the configuration was passed as a parameter.
        :param include_default:   Whether to include the default in addition to the provided plans. Default: True.
        :param default_kwargs:    Additional key word arguments to pass to the 'default' method of the given artifact
                                  library class.

        :return: The plans list.

        :raise MLRunInvalidArgumentError: If the plans were not passed in a list or a dictionary.
        """
        # Generate the available plans dictionary:
        available_plans = cls._get_library_plans()
        # if custom_plans is not None:
        #     available_plans = {**available_plans, **custom_plans}

        # Initialize the plans list:
        parsed_plans = []  # type: List[Plan]

        # Get the user input plans:
        artifacts_from_context = None
        if context is not None:
            artifacts_from_context = context.parameters.get(cls.CONTEXT_PARAMETER, None)
        for user_input in [artifacts, artifacts_from_context]:
            if user_input is not None:
                if isinstance(user_input, dict):
                    parsed_plans += cls._from_dict(
                        requested_plans=user_input, available_plans=available_plans
                    )
                elif isinstance(user_input, list):
                    parsed_plans += cls._from_list(
                        requested_plans=user_input, available_plans=available_plans
                    )
                else:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"Artifacts plans are expected to be given in a list or a dictionary, "
                        f"got: '{type(user_input)}'."
                    )

        # Get the library's default:
        if include_default:
            parsed_plans += cls.default(**default_kwargs)

        return parsed_plans

    @classmethod
    @abstractmethod
    def default(cls, **kwargs) -> list[Plan]:
        """
        Get the default artifacts plans list of this framework's library.

        :return: The default artifacts plans list.
        """
        pass

    @classmethod
    def _get_library_plans(cls) -> dict[str, type[Plan]]:
        """
        Get all the supported plans in this library.

        :return: The library's plans.
        """
        return {  # type: Dict[str, Type[Plan]]
            plan_name: plan_class
            for plan_name, plan_class in cls.__dict__.items()
            if isinstance(plan_class, type) and not plan_name.startswith("_")
        }

    @staticmethod
    def _from_dict(
        requested_plans: dict[str, dict], available_plans: dict[str, type[Plan]]
    ) -> list[Plan]:
        """
        Initialize a list of plans from a given configuration dictionary. The configuration is expected to be a
        dictionary of plans and their initialization parameters in the following format:

        {
            PLAN_NAME: {
                PARAMETER_NAME: PARAMETER_VALUE,
            },
        }

        :param requested_plans: The configurations of plans to initialize.
        :param available_plans: The available plans to initialize from.

        :return: The initialized plans list.

        :raise MLRunInvalidArgumentError: If the configuration was incorrect due to unsupported plan or miss use of
                                          parameters in the plan initializer.
        """
        # Go through the given configuration and initialize the plans accordingly:
        plans = []  # type: List[Plan]
        for plan_name, plan_parameters in requested_plans.items():
            # Validate the plan is in the library:
            if plan_name not in available_plans:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"The given artifact '{plan_name}' is not known in this artifacts library. The known artifacts "
                    f"are: {list(available_plans.keys())}."
                )
            # Try to create the plan with the given parameters:
            try:
                plans.append(available_plans[plan_name](**plan_parameters))
            except TypeError as error:
                # A TypeError was raised, that means there was a misuse of parameters in the plan's '__init__' method:
                raise mlrun.MLRunInvalidArgumentError(
                    f"The following artifact: '{plan_name}' cannot be parsed due to misuse of parameters: {error}"
                )

        return plans

    @staticmethod
    def _from_list(
        requested_plans: list[str], available_plans: dict[str, type[Plan]]
    ) -> list[Plan]:
        """
        Initialize a list of plans from a given configuration list. The configuration is expected to be a list of plans
        names to be initialized with their default configuration.

        :param requested_plans: The plans to initialize.
        :param available_plans: The available plans to initialize from.

        :return: The initialized plans list.

        :raise MLRunInvalidArgumentError: If the configuration was incorrect due to unsupported plan.
        """
        # Go through the given configuration and initialize the plans accordingly:
        plans = []  # type: List[Plan]
        for plan in requested_plans:
            # Initialized plan:
            if isinstance(plan, Plan):
                plans.append(plan)
            # Plan name that needed to be parsed:
            elif isinstance(plan, str):
                # Validate the plan is in the library:
                if plan not in available_plans:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"The given artifact '{plan}' is not known in this artifacts library. The known artifacts "
                        f"are: {list(available_plans.keys())}."
                    )
                # Create the plan and collect it:
                plans.append(available_plans[plan]())
            # Unsupported type:
            else:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Expecting a list of artifact plans or plans names from the artifacts library but given: "
                    f"'{type(plan)}'"
                )

        return plans
