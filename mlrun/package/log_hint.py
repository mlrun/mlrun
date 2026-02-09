import warnings
from typing import Any, Self

from pydantic import BaseModel, Field
from pydantic.config import ExtraValues

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

    itemized: bool | int = False
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

    @classmethod
    def model_validate(
        cls,
        obj: str | dict,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        """
        Override the default `model_validate` method to add support for parsing log hints from the old dictionary
        format.

        Note: This override is temporary and will be removed in MLRun 1.13.0, at which point only the new ``LogHint``
        format will be supported for parsing.

        :param obj:             The object to validate and parse into a LogHint instance. This can be in the old
                                dictionary format or the new LogHint format.
        :param strict:          Whether to perform strict validation. Passed to the superclass method.
        :param extra:           How to handle extra fields. Passed to the superclass method.
        :param from_attributes: Whether to populate the model from attributes. Passed to the superclass method
        :param context:         Additional context for validation. Passed to the superclass method.
        :param by_alias:        Whether to populate the model by alias. Passed to the superclass method
        :param by_name:         Whether to populate the model by field name. Passed to the superclass method.

        :return: An instance of ``LogHint`` created from the input object.
        """
        # CCheck if needed to construct from string:
        if isinstance(obj, str):
            return cls._from_string(log_hint_string=obj)

        # TODO: Remove in 1.13.0 - this method should only support parsing from the new LogHint format.
        # Check for the old dict format and raise a deprecation warning:
        if isinstance(obj, dict):
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

        return super().model_validate(
            obj=obj,
            strict=strict,
            extra=extra,
            from_attributes=from_attributes,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @classmethod
    def _from_string(cls, log_hint_string: str) -> "LogHint":
        """
        Create a LogHint object from a string. The string should be in the format of
        '<artifact_key> : <artifact_type>' or just '<artifact_key>'.

        :param log_hint_string: The log hint string to parse.

        :return: The created LogHint object.
        """
        # Check if only key is given:
        if ":" not in log_hint_string:
            key = log_hint_string
            artifact_type = None
        else:
            # Check for valid "<key> : <artifact type>" pattern:
            if log_hint_string.count(":") > 1:
                raise MLRunInvalidArgumentError(
                    f"Incorrect log hint pattern. Log hints can have only a single ':' in them to specify the "
                    f"desired artifact type the returned value will be logged as: "
                    f"'<artifact_key> : <artifact_type>', but given: {log_hint_string}"
                )
            # Split into key and type:
            key, artifact_type = log_hint_string.replace(" ", "").split(":")
            if key == "" or artifact_type == "":
                raise MLRunInvalidArgumentError(
                    f"Incorrect log hint pattern. The ':' in a log hint should specify the desired artifact type "
                    f"the returned value will be logged as in the following pattern: "
                    f"'<artifact_key> : <artifact_type>', but no key or artifact type was given: {log_hint_string}"
                )

        # Look for unbundle operator:
        key, itemized = cls._extract_unbundling_from_key(log_hint_key=key)

        return cls(
            key=key,
            artifact_type=artifact_type,
            itemized=itemized,
        )

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
