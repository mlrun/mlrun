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

import ast
import warnings
from typing import Any, Self

from pydantic.v1 import BaseModel, Field, validator

from mlrun.errors import MLRunInvalidArgumentError


class LogHint(BaseModel):
    """
    A log hint is a configuration to log an object returned from an MLRun function. Log hints are passed to the
    function's `run()` method via the `returns` argument.
    """

    key: str
    """
    The artifact key to log the object under.
    """

    tag: str = ""
    """
    The artifact tag to log the object under. Default is an empty string.
    """

    # TODO: Restore type to `bool | int` once migrated to Pydantic v2, which handles Union[bool, int]. Remove the
    #  `_validate_itemized` validator as well.
    itemized: Any = False
    """
    Determines if collections (lists or dicts) should be **unbundled** and logged as individual items.

    When `itemized` is enabled, the packager performs an **unbundling** process: instead of
    logging a collection as a single unit, it breaks it down into separate artifacts.
    Each item is logged under the primary key using either an index suffix (for sequences)
    or a sub-key suffix (for maps), inheriting the original log hint configuration.

    Accepts the following types:
    * `bool`:
        - `True`: Recursively **unbundles** the object all the way down.
        - `False` (default): Logs the collection as a single, opaque artifact.
    * `int`: Specifies the maximum depth of **unbundling**. For example, `1` will itemize the top-level collection but
      log nested collections as single units.
    """

    artifact_type: str | None = None
    """
    The artifact type to log the object as. If None is given, the default artifact type for the object's type will be
    used. Default is None.

    Common artifact types are listed in ``mlrun.package.ArtifactType``.
    """

    packing_kwargs: dict | None = Field(default_factory=dict)
    """
    Additional keyword arguments to pass to the packager's ``pack`` when packing the object for logging. To know which
    keyword arguments are supported, check the relevant packager (according to the returned object type) pack method
    (according to the given artifact type) documentation.
    """

    labels: dict[str, str] | None = None
    """
    Labels to add to the logged artifact.
    """

    extra_data: dict = Field(default_factory=dict)
    """
    Extra data to log alongside the artifact. To link to another package, write the key and a '...' as the value. For
    more information, see the 'Linking artifacts' section at the ``Packager`` or ``DefaultPackager`` documentation.
    """

    metrics: dict = Field(default_factory=dict)
    """
    Metrics to log alongside the model artifact (only for model artifacts). To link to another package, write the key
    and a '...' as the value. For more information, see the 'Linking artifacts' section at the ``Packager`` or
    ``DefaultPackager`` documentation.
    """

    @validator("itemized")
    def _validate_itemized(cls, v):  # noqa: N805
        if not isinstance(v, bool | int):
            raise ValueError(f"'itemized' must be bool or int, got {type(v).__name__}")
        return v

    @classmethod
    def parse_obj(cls, obj: Any) -> Self:
        """
        Override the default `model_validate` method to add support for parsing log hints from the old dictionary
        format.

        Note: This override is temporary and will be removed in MLRun 1.13.0, at which point only the new ``LogHint``
        format will be supported for parsing.

        :param obj: The object to validate and parse into a LogHint instance. This can be in the old dictionary format
                    or the new LogHint format.

        :return: An instance of ``LogHint`` created from the input object.
        """
        # TODO: Change to `model_validate` once Pydantic v2 is supported.
        # Check if needed to construct from string:
        if isinstance(obj, str):
            return cls._from_string(log_hint_string=obj)

        # TODO: Remove in 1.13.0 - this method should only support parsing from the new LogHint format.
        # Check for the old dict format and raise a deprecation warning:
        if isinstance(obj, dict):
            # Detect new LogHint-format dict (from .dict() serialization):
            if set(obj.keys()).issubset(
                set(cls.__fields__.keys())
            ) and "*" not in obj.get("key", ""):
                return super().parse_obj(obj=obj)
            # Potentially old format, try to parse and raise a warning:
            obj = obj.copy()
            key = obj.pop("key")
            key, itemized = cls._extract_unbundling_from_key(key)
            artifact_type = obj.pop("artifact_type", None)
            packing_kwargs = None
            if obj:
                # There are still some keys left in the dictionary, which means it's not following the new LogHint
                # format. Raise a warning:
                warnings.warn(
                    message=(
                        "Passing log hints as dictionaries will soon be deprecated (1.13.0). Please use the new "
                        "`mlrun.LogHint` class or use the string representation as before."
                    ),
                    category=FutureWarning,
                    stacklevel=2,
                )
                packing_kwargs = obj
            obj = {
                "key": key,
                "artifact_type": artifact_type,
                "itemized": itemized,
                "packing_kwargs": packing_kwargs,
            }

        return super().parse_obj(obj=obj)

    @classmethod
    def _from_string(cls, log_hint_string: str) -> "LogHint":
        """
        Create a LogHint object from a string. The string should be in the format of:

        * `<artifact_key>` - for a simple log hint with only the artifact key. Artifact key is mandatory.
        * `<unbundle_level>*<artifact_key>` or `*<artifact_key>` - to specify that the returned object should be
          itemized (unbundled and logged as separate items). The unbundle level is optional and can be an integer
          specifying the maximum depth of unbundling, or empty for full unbundling.
        * `<artifact_key> : <artifact_type>[<packing_kwarg1>=<value1>, <packing_kwarg2>=<value2>]` - to specify the
          artifact type. Artifact type is optional, but if given, the user can also specify packing kwargs in the same
          string. Packing kwargs are optional and should be given in the format of `<packing_kwarg>=<value>`, inside
          square brackets `[]` at the end of the string.

        :param log_hint_string: The log hint string to parse.

        :return: The created LogHint object.

        :raise MLRunInvalidArgumentError: If the log hint string has an incorrect pattern.
        """
        # Look for an artifact type:
        key, artifact_type = cls._extract_artifact_type_from_key(
            log_hint_key=log_hint_string
        )

        # Look for unbundle operator:
        key, itemized = cls._extract_unbundling_from_key(log_hint_key=key)

        # Look for packing kwargs in the log hint key and move them to the packing_kwargs field:
        if artifact_type:
            artifact_type, packing_kwargs = (
                cls._extract_packing_kwargs_from_artifact_type(
                    artifact_type=artifact_type
                )
            )
        else:
            packing_kwargs = {}

        return cls(
            key=key,
            artifact_type=artifact_type,
            itemized=itemized,
            packing_kwargs=packing_kwargs,
        )

    @staticmethod
    def _extract_artifact_type_from_key(log_hint_key: str) -> tuple[str, str | None]:
        """
        Extract artifact type information from a log hint key if exists. If the log hint key contains a colon ':', it
        indicates that an artifact type is specified. The part before the colon represents the actual artifact key,
        and the part after the colon is the artifact type.

        :param log_hint_key: The log hint key to extract artifact type information from.

        :return: A tuple containing the actual artifact key and the artifact type (or None if not specified).

        :raise MLRunInvalidArgumentError: If the log hint key has an incorrect pattern.
        """
        # Check if only key is given:
        if ":" not in log_hint_key:
            return log_hint_key.strip(), None

        # Check for valid "<key> : <artifact type>" pattern:
        if log_hint_key.count(":") > 1:
            raise MLRunInvalidArgumentError(
                f"Incorrect log hint pattern. Log hints can have only a single ':' in them to specify the "
                f"desired artifact type the returned value will be logged as: "
                f"'<artifact_key> : <artifact_type>', but given: {log_hint_key}"
            )

        # Split into key and type:
        key, artifact_type = log_hint_key.replace(" ", "").split(":")
        if key == "" or artifact_type == "":
            raise MLRunInvalidArgumentError(
                f"Incorrect log hint pattern. The ':' in a log hint should specify the desired artifact type "
                f"the returned value will be logged as in the following pattern: "
                f"'<artifact_key> : <artifact_type>', but no key or artifact type was given: {log_hint_key}"
            )

        return key.strip(), artifact_type.strip()

    @staticmethod
    def _extract_unbundling_from_key(log_hint_key: str) -> tuple[str, bool | int]:
        """
        Extract unbundling information from a log hint key if exists. If the log hint key contains an asterisk '*', it
        indicates that unbundling is required. The part before the asterisk represents the unbundle level (an integer or
        empty for full unbundling), and the part after the asterisk is the actual artifact key.

        :param log_hint_key: The log hint key to extract unbundling information from.

        :return: A tuple containing the actual artifact key and the unbundle level (True for full unbundling, False for
                 no unbundling, or an integer for specific unbundle level).
        """
        # Check if unbundling is required:
        if "*" not in log_hint_key:
            return log_hint_key, False

        # TODO: Remove in 1.13.0 - the '**' operator for dict unbundling is replaced by a single '*' operator:
        if "**" in log_hint_key:
            warnings.warn(
                message=(
                    "The '**' for packing dictionary items separately is replaced by a single '*', same as list. "
                    "Please read the documentation on the new bundling and unbundling feature. Using '**' will be "
                    "removed in MLRun 1.13.0. Currently replacing '**' with '*' automatically."
                ),
                category=FutureWarning,
                stacklevel=2,
            )
            log_hint_key = log_hint_key.replace("**", "*")

        # Extract unbundle level and key:
        unbundle_level, key = log_hint_key.split("*", 1)

        # Make sure a key is given:
        if not key.strip():
            raise MLRunInvalidArgumentError(
                f"Invalid log hint key '{log_hint_key}'. Key is missing after the unbundle operator '*' indicating "
                f"itemization. A log hint key with unbundling should be in the format of "
                f"'<unbundle_level>*<key>' or '*<key>' for full itemization."
            )

        # If unbundle level is given, convert to int:
        if unbundle_level.strip():
            try:
                unbundle_level = int(unbundle_level.strip())
            except ValueError:
                raise MLRunInvalidArgumentError(
                    f"Invalid unbundle level '{unbundle_level}' in log hint '{log_hint_key}'. "
                    f"Unbundle level must be an integer."
                )
        else:
            # If no level is given, set to True for full unbundling:
            unbundle_level = True

        return key.strip(), unbundle_level

    @staticmethod
    def _extract_packing_kwargs_from_artifact_type(
        artifact_type: str,
    ) -> tuple[str, dict]:
        """
        Extract packing kwargs from the artifact type string if exists. If the artifact type contains packing kwargs,
        they should be given in the format of '<artifact_type>[<packing_kwarg1>=<value1>, <packing_kwarg2>=<value2>]'.

        :param artifact_type: The artifact type string to extract packing kwargs from, or None if no artifact type was
                              specified.

        :return: A tuple containing the actual artifact type (or None) and a dictionary of packing kwargs.

        :raise MLRunInvalidArgumentError: If the artifact type string has an incorrect pattern.
        """
        # Check if packing kwargs are given:
        if "[" not in artifact_type:
            return artifact_type.strip(), {}

        # Check for valid pattern:
        if (
            not artifact_type.endswith("]")
            or artifact_type.count("[") > 1
            or artifact_type.count("]") > 1
        ):
            raise MLRunInvalidArgumentError(
                f"Incorrect log hint pattern for packing kwargs. Packing kwargs should be given in the format of "
                f"'<artifact_key> : <artifact_type>[<packing_kwarg1>=<value1>, <packing_kwarg2>=<value2>]', "
                f"but given: {artifact_type}"
            )

        # Extract packing kwargs string and convert to dictionary:
        open_bracket_index = artifact_type.index("[")
        close_bracket_index = artifact_type.index("]")
        packing_kwargs_string = artifact_type[
            open_bracket_index + 1 : close_bracket_index
        ]
        packing_kwargs = {}
        for kwarg in packing_kwargs_string.split(","):
            # Split to key and value:
            if "=" not in kwarg:
                raise MLRunInvalidArgumentError(
                    f"Incorrect log hint pattern for packing kwargs. Each packing kwarg should be given in the format "
                    f"'<packing_kwarg>=<value>', but given: {kwarg} in log hint: {artifact_type}"
                )
            kwarg_key, kwarg_value = kwarg.split("=", 1)
            # Try to convert the kwarg value to a Python literal:
            try:
                kwarg_value = ast.literal_eval(kwarg_value.strip())
            except Exception as eval_error:
                raise MLRunInvalidArgumentError(
                    f"The value for packing kwarg '{kwarg_key.strip()}' is not a valid Python literal value. Packing "
                    f"kwarg values should be valid Python literals (e.g. int, bool, list, None, or string with quotes)."
                ) from eval_error
            # Collect the kwarg:
            packing_kwargs[kwarg_key.strip()] = kwarg_value

        # Extract actual artifact type without packing kwargs:
        artifact_type = artifact_type[:open_bracket_index].strip()

        return artifact_type, packing_kwargs
