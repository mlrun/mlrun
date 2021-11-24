from typing import Dict, List, Optional

from .data_types import ValueType
from .model import ModelObj


class Entity(ModelObj):
    """data entity (index)"""

    def __init__(
        self,
        name: str = None,
        value_type: ValueType = None,
        description: str = None,
        labels: Optional[Dict[str, str]] = None,
    ):
        """data entity (index key)

        :param name:        entity name
        :param value_type:  type of the entity, e.g. ValueType.STRING, ValueType.INT
        :param description: test description of the entity
        :param labels:      a set of key/value labels (tags)
        """
        self.name = name
        self.description = description
        self.value_type = value_type
        if name and not value_type:
            self.value_type = ValueType.STRING
        self.labels = labels or {}


class Feature(ModelObj):
    """data feature"""

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
        value_type: str = None,
        dims: List[int] = None,
        description: str = None,
        aggregate: bool = None,
        name: str = None,
        validator=None,
        default: str = None,
        labels: Dict[str, str] = None,
    ):
        """data feature

        Features can be specified manually or inferred automatically (during ingest/preview)

        :param value_type:  type of the feature. Use the ValueType constants library e.g. ValueType.STRING,
                            ValueType.INT
        :param dims:        list of dimensions for vectors/tensors, e.g. [2, 2]
        :param description: text description of the feature
        :param aggregate:   is it an aggregated value
        :param name:        name of the feature
        :param validator:   feature validation policy
        :param default:     default value
        :param labels:      a set of key/value labels (tags)
        """
        self.name = name or ""
        self.value_type = value_type or ""
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


class Validator(ModelObj):
    """base validator"""

    kind = ""
    _dict_fields = ["kind", "check_type", "severity"]

    def __init__(self, check_type=None, severity=None):
        self._feature = None
        self.check_type = check_type
        self.severity = severity

    def set_feature(self, feature):
        self._feature = feature

    def check(self, value):
        return True, {}


class MinMaxValidator(Validator):
    """validate min/max value ranges"""

    kind = "minmax"
    _dict_fields = Validator._dict_fields + ["min", "max"]

    def __init__(self, check_type=None, severity=None, min=None, max=None):
        super().__init__(check_type, severity)
        self.min = min
        self.max = max

    def check(self, value):
        ok, args = super().check(value)
        if ok:
            if self.min is not None:
                if value < self.min:
                    return (
                        False,
                        {
                            "message": "value is smaller than min",
                            "min": self.min,
                            "value": value,
                        },
                    )
            if self.max is not None:
                if value > self.max:
                    return (
                        False,
                        {
                            "message": "value is greater than max",
                            "max": self.max,
                            "value": value,
                        },
                    )
        return ok, args


validator_kinds = {
    "": Validator,
    "minmax": MinMaxValidator,
}
