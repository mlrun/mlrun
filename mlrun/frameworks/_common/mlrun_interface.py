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
import copy
import functools
import inspect
from abc import ABC
from types import FunctionType, MethodType
from typing import Any, Generic, Optional, Union

from .utils import CommonTypes


class MLRunInterface(ABC, Generic[CommonTypes.MLRunInterfaceableType]):
    """
    An abstract class for enriching an object interface with the properties, methods and functions written below.

    A class inheriting MLRun interface should insert what ever it needs to be inserted to the object to the following
    attributes: '_PROPERTIES', '_METHODS' and '_FUNCTIONS'. Then it should implement 'add_interface' and call 'super'.

    In order to replace object's attributes, the attributes to replace are needed to be added to the attributes:
    '_REPLACED_PROPERTIES', '_REPLACED_METHODS' and '_REPLACED_FUNCTIONS'. The original attribute will be kept in a
    backup attribute with the prefix noted in '_ORIGINAL_ATTRIBUTE_NAME'. The replacing functions / methods will be
    located by looking for the prefix noted in '_REPLACING_ATTRIBUTE_NAME'. The replacing function / method can be an
    MLRunInterface class method that return a function / method.

    For example: if "x" is in the list then the method "object.x" will be stored as "object.original_x" and "object.x"
    will then point to the method "MLRunInterface.mlrun_x". If "mlrun_x" is a class method, it will point to the
    function returned from MLRunInterface.mlrun_x()".
    """

    # Attributes to be inserted so the MLRun interface will be fully enabled.
    _PROPERTIES = {}  # type: Dict[str, Any]
    _METHODS = []  # type: List[str]
    _FUNCTIONS = []  # type: List[str]

    # Attributes to replace so the MLRun interface will be fully enabled.
    _REPLACED_PROPERTIES = {}  # type: Dict[str, Any]
    _REPLACED_METHODS = []  # type: List[str]
    _REPLACED_FUNCTIONS = []  # type: List[str]

    # TODO: Add _OPTIONALLY_REPLACED_PROPERTIES, _OPTIONALLY_REPLACED_METHODS and _OPTIONALLY_REPLACED_FUNCTIONS

    # Name template for the replaced attribute to be stored as in the object.
    _ORIGINAL_ATTRIBUTE_NAME = "original_{}"

    # Name template of the function / method to look for to replace the original function / method.
    _REPLACING_ATTRIBUTE_NAME = "mlrun_{}"

    @classmethod
    def add_interface(
        cls,
        obj: CommonTypes.MLRunInterfaceableType,
        restoration: CommonTypes.MLRunInterfaceRestorationType = None,
    ):
        """
        Enrich the object with this interface properties, methods and functions so it will have this framework MLRun's
        features.

        :param obj:         The object to enrich his interface.
        :param restoration: Restoration information tuple as returned from 'remove_interface' in order to add the
                            interface in a certain state.
        """
        # Set default value to the restoration data:
        if restoration is None:
            restoration = (None, None, None)

        # Add the MLRun properties:
        cls._insert_properties(
            obj=obj,
            properties=restoration[0],
        )

        # Replace the object's properties in MLRun's properties:
        cls._replace_properties(obj=obj, properties=restoration[1])

        # Add the MLRun functions:
        cls._insert_functions(obj=obj)

        # Replace the object's functions / methods in MLRun's functions / methods:
        cls._replace_functions(obj=obj, functions=restoration[2])

    @classmethod
    def remove_interface(
        cls, obj: CommonTypes.MLRunInterfaceableType
    ) -> CommonTypes.MLRunInterfaceRestorationType:
        """
        Remove the MLRun features from the given object. The properties and replaced attributes found in the object will
        be returned.

        :param obj: The object to remove the interface from.

        :return: A tuple of interface restoration information:
                 [0] = The interface properties.
                 [1] = The replaced properties.
                 [2] = The replaced methods and functions.
        """
        # Get the interface properties from the element:
        properties = {
            attribute: getattr(obj, attribute)
            for attribute in cls._PROPERTIES
            if hasattr(obj, attribute)  # Later it will be asserted.
        }

        # Get the replaced properties from the object:
        replaced_properties = {
            attribute: getattr(obj, attribute)
            for attribute in cls._REPLACED_PROPERTIES
            if hasattr(obj, cls._ORIGINAL_ATTRIBUTE_NAME.format(attribute))
        }

        # Get the replaced methods and functions from the object:
        replaced_functions = [
            function_name
            for function_name in [*cls._REPLACED_METHODS, *cls._REPLACED_FUNCTIONS]
            if hasattr(obj, cls._ORIGINAL_ATTRIBUTE_NAME.format(function_name))
        ]

        # Restore the replaced attributes:
        for attribute_name in [
            *replaced_properties,
            *replaced_functions,
        ]:
            cls._restore_attribute(obj=obj, attribute_name=attribute_name)

        # Remove the interface from the object:
        for attribute_name in [*cls._PROPERTIES, *cls._METHODS, *cls._FUNCTIONS]:
            assert hasattr(
                obj, attribute_name
            ), f"Can't remove the attribute '{attribute_name}' as the object doesn't has it."
            # Mark it first as None so the actual object won't be deleted:
            setattr(obj, attribute_name, None)
            delattr(obj, attribute_name)

        return properties, replaced_properties, replaced_functions

    @classmethod
    def is_applied(cls, obj: CommonTypes.MLRunInterfaceableType) -> bool:
        """
        Check if the given object has MLRun interface attributes in it. Interface is applied if all of its attributes
        are found in the object. If only replaced attributes are configured in the interface, then the interface is
        applied if some of at least one is found in the object.

        :param obj: The object to check.

        :return: True if the MLRun interface is applied on the object and False if not.
        """
        # Check for the attributes:
        attributes = [*cls._PROPERTIES, *cls._METHODS, *cls._FUNCTIONS]
        if attributes:
            return all(hasattr(obj, attribute) for attribute in attributes)

        # The interface has only replaced attributes, check if at least one is in the object:
        replaced_attributes = [
            *cls._REPLACED_PROPERTIES,
            *cls._REPLACED_METHODS,
            *cls._REPLACED_FUNCTIONS,
        ]
        return any(hasattr(obj, attribute) for attribute in replaced_attributes)

    @classmethod
    def _insert_properties(
        cls,
        obj: CommonTypes.MLRunInterfaceableType,
        properties: Optional[dict[str, Any]] = None,
    ):
        """
        Insert the properties of the interface to the object. The properties default values are being copied (not deep
        copied) into the object.

        :param obj:        The object to enrich.
        :param properties: Properties to set in the object.
        """
        # Set default dictionary if there is no properties to restore:
        if properties is None:
            properties = {}

        # Verify the provided properties are supported by this interface (noted in the interface '_PROPERTIES'):
        error_properties = [
            property_name
            for property_name in properties
            if property_name not in cls._PROPERTIES
        ]
        assert not error_properties, (
            f"The following properties provided to insert to the object are not supported by this interface: "
            f"{error_properties}"
        )

        # Insert the properties, copy only default values:
        for property_name, default_value in cls._PROPERTIES.items():
            # Verify there is no property with the same name in the object:
            assert not hasattr(obj, property_name), (
                f"Can't insert the property '{property_name}' as the object already have an attribute with the same "
                f"name."
            )
            # Insert the property to the object prioritizing given values over default ones:
            value = (
                properties[property_name]
                if property_name in properties
                else copy.copy(default_value)
            )
            setattr(obj, property_name, value)

    @classmethod
    def _insert_functions(cls, obj: CommonTypes.MLRunInterfaceableType):
        """
        Insert the functions / methods of the interface to the object.

        :param obj: The object to enrich.
        """
        # Insert the functions / methods:
        for function_name in [*cls._METHODS, *cls._FUNCTIONS]:
            # Verify there is no function / method with the same name in the object:
            assert not hasattr(obj, function_name), (
                f"Can't insert the function / method '{function_name}' as the object already have a function / method "
                f"with the same name. To replace a function / method, add the name of the function / method to the "
                f"'_REPLACED_METHODS' / '_REPLACED_METHODS' list and follow the instructions documented."
            )
            # Get the function / method:
            func = getattr(cls, function_name)
            # If the function is a method and not a function (appears in '_METHODS' and not '_FUNCTIONS'), set the
            # 'self' to the object:
            if function_name in cls._METHODS:
                func = MethodType(func, obj)
            # Insert the function / method to the object:
            setattr(obj, function_name, func)

    @classmethod
    def _replace_properties(
        cls,
        obj: CommonTypes.MLRunInterfaceableType,
        properties: Optional[dict[str, Any]] = None,
    ):
        """
        Replace the properties of the given object according to the configuration in the MLRun interface.

        :param obj:        The object to replace its properties.
        :param properties: The properties to replace in the object. Default: all the properties in the interface
                           '_REPLACE_PROPERTIES' dictionary.
        """
        # Set default replacing properties if there are no properties given:
        if properties is None:
            properties = {
                property_name: copy.copy(property_value)
                for property_name, property_value in cls._REPLACED_PROPERTIES.items()
            }
        else:
            # Verify the provided properties are supported by this interface (noted in the interface's
            # '_REPLACED_PROPERTIES' dictionary):
            error_properties = [
                property_name
                for property_name in properties
                if property_name not in cls._REPLACED_PROPERTIES
            ]
            assert not error_properties, (
                f"The following properties provided to be replace in the object are not supported by this "
                f"interface: {error_properties}"
            )

        # Replace the properties in the object:
        for property_name, property_value in properties.items():
            # Verify there is a property with this name in the object to replace:
            assert hasattr(
                obj, property_name
            ), f"Can't replace the property '{property_name}' as the object doesn't have a property with this name."
            # Replace the property:
            cls._replace_property(
                obj=obj,
                property_name=property_name,
                property_value=property_value,
                include_none=True,
            )

    @classmethod
    def _replace_functions(
        cls,
        obj: CommonTypes.MLRunInterfaceableType,
        functions: Optional[list[str]] = None,
    ):
        """
        Replace the functions / methods of the given object according to the configuration in the MLRun interface.

        :param obj:       The object to replace its functions / methods.
        :param functions: The functions / methods to replace in the object. Default: all the functions / methods in
                          the interface '_REPLACE_METHODS' and '_REPLACE_FUNCTIONS' lists.
        """
        # Set default list if there are no functions / methods to restore:
        if functions is None:
            functions = [*cls._REPLACED_METHODS, *cls._REPLACED_FUNCTIONS]
        else:
            # Verify the provided functions / methods are supported by this interface (noted in the interface's
            # '_REPLACED_METHODS' or '_REPLACE_FUNCTIONS' lists):
            error_functions = [
                function_name
                for function_name in functions
                if function_name
                not in [*cls._REPLACED_METHODS, *cls._REPLACED_FUNCTIONS]
            ]
            assert not error_functions, (
                f"The following functions / methods provided to be replace in the object are not supported by this "
                f"interface: {error_functions}"
            )

        # Replace the functions / methods in the object:
        for function_name in functions:
            # Verify there is a function / method with this name in the object to replace:
            assert hasattr(obj, function_name), (
                f"Can't replace the function / method '{function_name}' as the object doesn't have a function / method "
                f"with this name."
            )
            # Replace the method:
            cls._replace_function(obj=obj, function_name=function_name)

    @classmethod
    def _replace_property(
        cls,
        obj: CommonTypes.MLRunInterfaceableType,
        property_name: str,
        property_value: Any = None,
        include_none: bool = False,
    ):
        """
        Replace the property in the object with the configured property in this interface. The original property will be
        stored in a backup attribute with the prefix noted in '_ORIGINAL_ATTRIBUTE_NAME' and the replacing property
        will be the one with the prefix noted in '_REPLACING_ATTRIBUTE_NAME'. If the property value should be None, set
        'include_none' to True, otherwise the interface default will be copied if 'property_value' is None.

        :param obj:            The object to replace its property.
        :param property_name:  The property name to replace.
        :param property_value: The value to set. If not provided, the interface's default will be copied.
        :param include_none:   Whether to enable the property value to be None.
        """
        # Get the original property from the object:
        original_property = getattr(obj, property_name)

        # Set a backup attribute with for the original property:
        original_property_name = cls._ORIGINAL_ATTRIBUTE_NAME.format(property_name)
        setattr(obj, original_property_name, original_property)

        # Check if a value is provided, if not copy the default value in this interface if None should not be included:
        if not include_none and property_value is None:
            property_value = copy.copy(cls._REPLACED_PROPERTIES[property_name])

        # Replace the property:
        setattr(obj, property_name, property_value)

    @classmethod
    def _replace_function(
        cls, obj: CommonTypes.MLRunInterfaceableType, function_name: str
    ):
        """
        Replace the method / function in the object with the configured method / function in this interface. The
        original method / function will be stored in a backup attribute with the prefix noted in
        '_ORIGINAL_ATTRIBUTE_NAME' and the replacing method / function will be the one with the prefix noted in
        '_REPLACING_ATTRIBUTE_NAME'.

        :param obj:           The object to replace its method.
        :param function_name: The method / function name to replace.
        """
        # Get the original function from the object:
        original_function = getattr(obj, function_name)

        # Set a backup attribute with for the original function:
        original_function_name = cls._ORIGINAL_ATTRIBUTE_NAME.format(function_name)
        setattr(obj, original_function_name, original_function)

        # Get the function to replace from the interface:
        replacing_function_name = cls._REPLACING_ATTRIBUTE_NAME.format(function_name)
        replacing_function = getattr(cls, replacing_function_name)

        # Check if the replacing function is a class method (returning a function to use as the replacing function):
        if inspect.ismethod(replacing_function):
            replacing_function = replacing_function()

        # Wrap the replacing function with 'functools.wraps' decorator so the properties of the original function will
        # be passed to the replacing function:
        replacing_function = functools.wraps(original_function)(replacing_function)

        # If the replacing function is a method and not a function (appears in the _REPLACED_METHODS and not
        # _REPLACED_FUNCTIONS), set the 'self' to the object:
        if function_name in cls._REPLACED_METHODS:
            replacing_function = MethodType(replacing_function, obj)

        # Replace the function:
        setattr(obj, function_name, replacing_function)

    @classmethod
    def _restore_attribute(
        cls, obj: CommonTypes.MLRunInterfaceableType, attribute_name: str
    ):
        """
        Restore the replaced attribute (property, method or function) in the object, removing the backup attribute as
        well.

        :param obj:            The object to restore its method.
        :param attribute_name: The method to restore.
        """
        # Get the original attribute:
        original_attribute_name = cls._ORIGINAL_ATTRIBUTE_NAME.format(attribute_name)
        original_attribute = getattr(obj, original_attribute_name)

        # Set the attribute to point back to the original attribute:
        setattr(obj, attribute_name, original_attribute)

        # Remove the original backup attribute:
        setattr(obj, original_attribute_name, None)
        delattr(obj, original_attribute_name)

    @staticmethod
    def _get_function_argument(
        func: FunctionType,
        argument_name: str,
        passed_args: Optional[tuple] = None,
        passed_kwargs: Optional[dict] = None,
        default_value: Any = None,
    ) -> tuple[Any, Union[str, int, None]]:
        """
        Get a passed argument (from *args or **kwargs) to a function. If the argument was not found the default value
        will be returned. In addition, the keyword of the argument in `kwargs` or the index of the argument in `args`
        will be returned as well.

        :param func:          The function that is being called.
        :param argument_name: The argument name to get.
        :param passed_args:   The passed arguments to the function (*args).
        :param passed_kwargs: The passed keyword arguments to the function (*kwargs).
        :param default_value: The default value to use in case it was not passed.

        :return: A tuple of:
                 [0] = The argument value or the default value if it was not found in any of the arguments.
                 [1] = If it was found in `kwargs` - the keyword of the argument. If it was found in `args` - the index
                       of the argument. If it was not found, None.
        """
        # Set default values for arguments data structures:
        if passed_args is None:
            passed_args = []
        if passed_kwargs is None:
            passed_kwargs = {}

        # Check in the key word arguments first:
        if argument_name in passed_kwargs:
            return passed_kwargs[argument_name], argument_name

        # Check in the arguments, inspecting the function's parameters to get the right index:
        func_parameters = {
            parameter_name: i
            for i, parameter_name in enumerate(
                inspect.signature(func).parameters.keys()
            )
        }
        if (
            argument_name in func_parameters
            and len(passed_args) >= func_parameters[argument_name] + 1
        ):
            return (
                passed_args[func_parameters[argument_name]],
                func_parameters[argument_name],
            )

        # The argument name was not found:
        return default_value, None
