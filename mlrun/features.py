from typing import Optional, Dict
import pandas as pd
import numpy as np

from pandas.io.json._table_schema import convert_pandas_type_to_json_field
from mlrun.data_types import pd_schema_to_value_type, ValueType
from mlrun.model import ModelObj


default_num_bins = 20


class Entity(ModelObj):
    """data entity (index)"""

    def __init__(
        self,
        name: str = None,
        value_type: ValueType = None,
        description: str = None,
        labels: Optional[Dict[str, str]] = None,
    ):
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
    ]

    def __init__(
        self,
        value_type: ValueType = None,
        description=None,
        aggregate=None,
        name=None,
        validator=None,
        default=None,
        labels: Dict[str, str] = None,
    ):
        self.name = name or ""
        self.value_type: ValueType = value_type or ""
        self.dims = None
        self.description = description
        self.default = default
        self.labels = labels or {}
        self.aggregate = aggregate
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


class InferOptions:
    Null = 0
    Entities = 1
    Features = 2
    Index = 4
    Stats = 8
    Histogram = 16
    Preview = 32

    @staticmethod
    def schema():
        return InferOptions.Entities + InferOptions.Features + InferOptions.Index

    @staticmethod
    def all_stats():
        return InferOptions.Stats + InferOptions.Histogram + InferOptions.Preview

    @staticmethod
    def all():
        return (
            InferOptions.schema()
            + InferOptions.Stats
            + InferOptions.Histogram
            + InferOptions.Preview
        )

    @staticmethod
    def default():
        return InferOptions.all()

    @staticmethod
    def get_common_options(one, two):
        return one & two


def infer_schema_from_df(
    df: pd.DataFrame,
    features,
    entities,
    timestamp_key: str = None,
    entity_columns=None,
    options: InferOptions = InferOptions.Null,
):
    """infer feature set schema from dataframe"""
    timestamp_fields = []
    current_entities = list(entities.keys())
    entity_columns = entity_columns or []

    def upsert_entity(name, value_type):
        if name in current_entities:
            entities[name].value_type = value_type
        else:
            entities[name] = Entity(name=column, value_type=value_type)

    for column, series in df.items():
        value_type = _get_column_type(series)
        is_entity = column in entity_columns or column in current_entities
        if is_entity:
            upsert_entity(column, value_type)
        elif (
            InferOptions.get_common_options(options, InferOptions.Features)
            and column != timestamp_key
        ):
            if column in features.keys():
                features[column].value_type = value_type
            else:
                features[column] = Feature(name=column, value_type=value_type)
        if value_type == "datetime" and not is_entity:
            timestamp_fields.append(column)

    if InferOptions.get_common_options(options, InferOptions.Index):
        # infer types of index fields
        if df.index.name:
            value_type = _get_column_type(df.index)
            upsert_entity(df.index.name, value_type)
        elif df.index.nlevels > 1:
            for level, name in zip(df.index.levels, df.index.names):
                value_type = _get_column_type(level)
                upsert_entity(name, value_type)
                if value_type == "datetime":
                    timestamp_fields.append(name)

    if len(timestamp_fields) == 1 and not timestamp_key:
        return timestamp_fields[0]
    return timestamp_key


def _get_column_type(column):
    field = convert_pandas_type_to_json_field(column)
    return pd_schema_to_value_type(field["type"])


def get_df_stats(df, options, num_bins=None):
    """get per column data stats from dataframe"""

    results_dict = {}
    num_bins = num_bins or default_num_bins
    if InferOptions.get_common_options(options, InferOptions.Index) and df.index.name:
        df = df.reset_index()
    for col, values in df.describe(
        include="all", percentiles=[], datetime_is_numeric=True
    ).items():
        stats_dict = {}
        for stat, val in values.dropna().items():
            if stat != "50%":
                if isinstance(val, (float, np.floating, np.float64)):
                    stats_dict[stat] = float(val)
                elif isinstance(val, (int, np.integer, np.int64)):
                    stats_dict[stat] = int(val)
                else:
                    stats_dict[stat] = str(val)

        if InferOptions.get_common_options(
            options, InferOptions.Histogram
        ) and pd.api.types.is_numeric_dtype(df[col]):
            # store histogram
            try:
                hist, bins = np.histogram(df[col], bins=num_bins)
                stats_dict["hist"] = [hist.tolist(), bins.tolist()]
            except Exception:
                pass

        results_dict[col] = stats_dict
    return results_dict


def get_df_preview(df, preview_lines=20):
    """capture preview data from df"""
    # record sample rows from the dataframe
    length = df.shape[0]
    shortdf = df
    if length > preview_lines:
        shortdf = df.head(preview_lines)
    return [shortdf.columns.values.tolist()] + shortdf.values.tolist()


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
