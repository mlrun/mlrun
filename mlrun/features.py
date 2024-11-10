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
import math
import re
from typing import Optional, Union

from .data_types import ValueType, python_type_to_value_type
from .errors import MLRunRuntimeError, err_to_str
from .model import ModelObj


def _limited_string(value: str, max_size: int = 40):
    """
    Provide limited string size, typically for reporting original value
    in case of error (and for better identification of error location
    based on presenting part of original value)
    """
    return (
        value
        if (value is None) or (len(value) <= max_size)
        else value[:max_size] + "..."
    )


class Entity(ModelObj):
    """data entity (index)"""

    kind = "entity"

    def __init__(
        self,
        name: Optional[str] = None,
        value_type: Union[ValueType, str] = None,
        description: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
    ):
        """data entity (index key)

        :param name:        entity name
        :param value_type:  type of the entity, e.g. ValueType.STRING, ValueType.INT (default ValueType.STRING)
        :param description: test description of the entity
        :param labels:      a set of key/value labels (tags)
        """
        self.name = name
        self.description = description
        self.value_type = ValueType(value_type) if value_type else None
        if name and not value_type:
            self.value_type = ValueType.STRING
        self.labels = labels or {}

    def __eq__(self, other):
        return self.name == other.name


class Feature(ModelObj):
    _dict_fields = [
        "name",
        "description",
        "value_type",
        "dims",
        "default",
        "labels",
        "aggregate",
        "validator",
        "origin",
    ]

    def __init__(
        self,
        value_type: Union[ValueType, str] = None,
        dims: Optional[list[int]] = None,
        description: Optional[str] = None,
        aggregate: Optional[bool] = None,
        name: Optional[str] = None,
        validator=None,
        default: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
    ):
        """data feature

        Features can be specified manually or inferred automatically (during ingest/preview)

        :param value_type:  type of the feature. Use the ValueType constants library e.g. ValueType.STRING,
                            ValueType.INT (default ValueType.STRING)
        :param dims:        list of dimensions for vectors/tensors, e.g. [2, 2]
        :param description: text description of the feature
        :param aggregate:   is it an aggregated value
        :param name:        name of the feature
        :param validator:   feature validation policy
        :param default:     default value
        :param labels:      a set of key/value labels (tags). Labels can be used to filter featues, for example,
                            in the UI Feature store page.
        """
        self.name = name or ""
        if isinstance(value_type, ValueType):
            self.value_type = value_type
        elif value_type is not None:
            self.value_type = python_type_to_value_type(value_type)
        else:
            self.value_type = ValueType.STRING
        self.dims = dims
        self.description = description
        self.default = default
        self.labels = labels or {}
        self.aggregate = aggregate
        self.origin = None  # used to link the feature to the feature set origin (inside vector.status)
        self._validator = validator

    @property
    def validator(self):
        return self._validator

    @validator.setter
    def validator(self, validator):
        if isinstance(validator, dict):
            kind = validator.get("kind")
            validator = validator_kinds[kind].from_dict(validator)
        self._validator = validator


class BasicTypeValidator:
    def __init__(self):
        pass

    def check(self, value_type, value):
        return True, {}


class ConvertTypeValidator(BasicTypeValidator):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def check(self, value_type, value):
        ok, args = super().check(value_type, value)
        if ok:
            try:
                self.func(value)
            except Exception as err:
                return (
                    False,
                    {"message": err_to_str(err), "type": value_type},
                )
        return ok, args


class RangeTypeValidator(BasicTypeValidator):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max

    def check(self, value_type, value):
        ok, args = super().check(value_type, value)
        if ok:
            try:
                if value < self.min:
                    return (
                        False,
                        {
                            "message": "Value is smaller than min range",
                            "type": value_type,
                            "min range": self.min,
                            "value": _limited_string(value),
                        },
                    )
                if value > self.max:
                    return (
                        False,
                        {
                            "message": "Value is greater than max range",
                            "type": value_type,
                            "max range": self.max,
                            "value": _limited_string(value),
                        },
                    )
            except Exception as err:
                return (
                    False,
                    {"message": err_to_str(err), "type": value_type},
                )

        return ok, args


# TODO: add addition validation for commented types
type_validator = {
    #   ValueType.BOOL: it does not make sense to do validation for BOOL (everything is True or False by default)
    ValueType.INT8: RangeTypeValidator(-128, 127),
    ValueType.INT16: RangeTypeValidator(-32768, 32767),
    ValueType.INT32: RangeTypeValidator(-2147483648, 2147483647),
    ValueType.INT64: RangeTypeValidator(-9223372036854775808, 9223372036854775807),
    ValueType.INT128: RangeTypeValidator(-math.pow(2, 127), math.pow(2, 127) - 1),
    ValueType.UINT8: RangeTypeValidator(0, 255),
    ValueType.UINT16: RangeTypeValidator(0, 65535),
    ValueType.UINT32: RangeTypeValidator(0, 4294967295),
    ValueType.UINT64: RangeTypeValidator(0, 18446744073709551615),
    ValueType.UINT128: RangeTypeValidator(0, math.pow(2, 128)),
    #   ValueType.FLOAT16: None,
    ValueType.FLOAT: ConvertTypeValidator(float),
    ValueType.DOUBLE: ConvertTypeValidator(float),
    #   ValueType.BFLOAT16: None,
    ValueType.BYTES: ConvertTypeValidator(bytes),
    #   ValueType.STRING: it does not make sense to do validation for STRING (everything is valid also '\x00', '\xff')
    #   ValueType.DATETIME: None,
    #   ValueType.BYTES_LIST: None,
    #   ValueType.STRING_LIST: None,
    #   ValueType.INT32_LIST: None,
    #   ValueType.INT64_LIST: None,
    #   ValueType.DOUBLE_LIST: None,
    #   ValueType.FLOAT_LIST: None,
    #   ValueType.BOOL_LIST: None,
}


class Validator(ModelObj):
    """Base validator"""

    kind = ""
    _dict_fields = ["kind", "check_type", "severity"]

    def __init__(
        self, check_type: Optional[bool] = None, severity: Optional[str] = None
    ):
        """Base validator

        example::

            from mlrun.features import Validator

            # Add validator to the feature 'bid' with check type
            quotes_set["bid"].validator = Validator(check_type=True, severity="info")

        :param check_type:  check feature type e.g. True, False
        :param severity:    severity name e.g. info, warning, etc.
        """
        self._feature = None
        self.check_type = check_type
        self.severity = severity

    def set_feature(self, feature: Feature):
        self._feature = feature

    def check(self, value):
        if self.check_type:
            if self._feature.value_type is not None:
                if self._feature.value_type in type_validator:
                    return type_validator[self._feature.value_type].check(
                        self._feature.value_type, value
                    )
        return True, {}


class MinMaxValidator(Validator):
    """Validate min/max value ranges"""

    kind = "minmax"
    _dict_fields = Validator._dict_fields + ["min", "max"]

    def __init__(
        self,
        check_type: Optional[bool] = None,
        severity: Optional[str] = None,
        min=None,
        max=None,
    ):
        """Validate min/max value ranges

        example::

            from mlrun.features import MinMaxValidator

            # Add validator to the feature 'bid', where valid
            # minimal value is 52
            quotes_set["bid"].validator = MinMaxValidator(min=52, severity="info")

        :param check_type:  check feature type e.g. True, False
        :param severity:    severity name e.g. info, warning, etc.
        :param min:         minimal valid size
        :param max:         maximal valid size
        """
        super().__init__(check_type, severity)
        self.min = min
        self.max = max

    def check(self, value):
        ok, args = super().check(value)
        if ok:
            try:
                if self.min is not None:
                    if value < self.min:
                        return (
                            False,
                            {
                                "message": "value is smaller than min",
                                "min": self.min,
                                "value": _limited_string(str(value)),
                            },
                        )
                if self.max is not None:
                    if value > self.max:
                        return (
                            False,
                            {
                                "message": "value is greater than max",
                                "max": self.max,
                                "value": _limited_string(str(value)),
                            },
                        )
            except Exception as err:
                return (
                    False,
                    {"message": err_to_str(err), "type": self.kind},
                )
        return ok, args


class MinMaxLenValidator(Validator):
    """Validate min/max length value ranges"""

    kind = "minmaxlen"
    _dict_fields = Validator._dict_fields + ["min", "max"]

    def __init__(
        self,
        check_type: Optional[bool] = None,
        severity: Optional[str] = None,
        min=None,
        max=None,
    ):
        """Validate min/max length value ranges

        example::

            from mlrun.features import MinMaxLenValidator

            # Add length validator to the feature 'ticker', where valid
            # minimal length is 1 and maximal length is 10
            quotes_set["ticker"].validator = MinMaxLenValidator(
                min=1, max=10, severity="info"
            )

        :param check_type:  check feature type e.g. True, False
        :param severity:    severity name e.g. info, warning, etc.
        :param min:         minimal valid length size
        :param max:         maximal valid length size
        """
        super().__init__(check_type, severity)
        self.min = min
        self.max = max

    def check(self, value):
        ok, args = super().check(value)
        if ok:
            try:
                if self.min is not None:
                    if len(value) < self.min:
                        return (
                            False,
                            {
                                "message": "Length value is smaller than min",
                                "min": self.min,
                                "length value": len(value),
                            },
                        )
                if self.max is not None:
                    if len(value) > self.max:
                        return (
                            False,
                            {
                                "message": "Length value is greater than max",
                                "max": self.max,
                                "length value": len(value),
                            },
                        )
            except Exception as err:
                return (
                    False,
                    {"message": err_to_str(err), "type": self.kind},
                )

        return ok, args


class RegexValidator(Validator):
    """Validate value based on regular expression"""

    kind = "regex"
    _dict_fields = Validator._dict_fields + ["regex"]

    def __init__(
        self,
        check_type: Optional[bool] = None,
        severity: Optional[str] = None,
        regex=None,
    ):
        """Validate value based on regular expression

        example::

            from mlrun.features import RegexValidator

            # Add regular expression validator to the feature 'name' and
            # expression '(\b[A-Za-z]{1}[0-9]{7}\b)' where valid values are
            # e.g. A1234567, z9874563, etc.
            quotes_set["name"].validator = RegexValidator(
                regex=r"(\b[A-Za-z]{1}[0-9]{7}\b)", severity="info"
            )

        :param check_type:  check feature type e.g. True, False
        :param severity:    severity name e.g. info, warning, etc.
        :param regex:       regular expression for validation
        """
        super().__init__(check_type, severity)
        self.regex = regex
        self.regex_compile = re.compile(self.regex) if self.regex else None

    def check(self, value):
        ok, args = super().check(value)
        if ok:
            try:
                if self.regex is not None:
                    if not re.fullmatch(self.regex_compile, value):
                        return (
                            False,
                            {
                                "message": "Value is not valid with regular expression",
                                "regexp": self.regex,
                                "value": _limited_string(str(value)),
                            },
                        )
            except Exception as err:
                return (
                    False,
                    {"message": err_to_str(err), "type": self.kind},
                )
        return ok, args

    @classmethod
    def from_dict(
        cls, struct=None, fields=None, deprecated_fields: Optional[dict] = None
    ):
        new_obj = super().from_dict(
            struct=struct, fields=fields, deprecated_fields=deprecated_fields
        )
        if hasattr(new_obj, "regex"):
            new_obj.regex_compile = re.compile(new_obj.regex) if new_obj.regex else None
        else:
            raise MLRunRuntimeError(
                f"Object with type {type(new_obj)} "
                f"have to contain `regex` attribute"
            )
        return new_obj


validator_kinds = {
    "": Validator,
    "minmax": MinMaxValidator,
    "minmaxlen": MinMaxLenValidator,
    "regex": RegexValidator,
}
