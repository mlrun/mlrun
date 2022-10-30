# Copyright 2018 Iguazio
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
import re
import uuid
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from storey import MapClass

import mlrun.errors
from mlrun.serving.server import get_event_time
from mlrun.serving.utils import StepToDict
from mlrun.utils import get_in


def get_engine(first_event):
    if hasattr(first_event, "body"):
        first_event = first_event.body
    if isinstance(first_event, pd.DataFrame):
        return "pandas"
    return "storey"


class MLRunStep(MapClass):
    def __init__(self, **kwargs):
        """Abstract class for mlrun step.
        Can be used in pandas/storey feature set ingestion"""
        super().__init__(**kwargs)

    def do(self, event):
        """
        This method defines the do method of this class according to the first event type.
        """
        engine = get_engine(event)
        if engine == "pandas":
            self.do = self._do_pandas
        else:
            self.do = self._do_storey
        return self.do(event)

    def _do_pandas(self, event):
        raise NotImplementedError

    def _do_storey(self, event):
        raise NotImplementedError


class FeaturesetValidator(StepToDict, MLRunStep):
    """Validate feature values according to the feature set validation policy"""

    def __init__(self, featureset=None, columns=None, name=None, **kwargs):
        """Validate feature values according to the feature set validation policy

        :param featureset: feature set uri (or "." for current feature set pipeline)
        :param columns:    names of the columns/fields to validate
        :param name:       step name
        :param kwargs:     optional kwargs (for storey)
        """
        kwargs["full_event"] = True
        super().__init__(**kwargs)
        self._validators = {}
        self.featureset = featureset or "."
        self.columns = columns
        self.name = name
        if not self.context:
            return
        self._featureset = self.context.get_store_resource(featureset)
        for key, feature in self._featureset.spec.features.items():
            if feature.validator and (not columns or key in columns):
                feature.validator.set_feature(feature)
                self._validators[key] = feature.validator

    def _do_storey(self, event):
        body = event.body
        for name, validator in self._validators.items():
            if name in body:
                ok, args = validator.check(body[name])
                if not ok:
                    message = args.pop("message")
                    key_text = f" key={event.key}" if event.key else ""
                    if event.time:
                        key_text += f" time={event.time}"
                    print(
                        f"{validator.severity}! {name} {message},{key_text} args={args}"
                    )
        return event

    def _do_pandas(self, event):
        body = event.body
        for column in body:
            validator = self._validators.get(column, None)
            if validator:
                violations = 0
                all_args = []
                for i in body.index:
                    # check each body entry if there is validator for it
                    ok, args = validator.check(body.at[i, column])
                    if not ok:
                        violations += 1
                        all_args.append(args)
                        message = args.pop("message")
                if violations != 0:
                    text = f" column={column}, has {violations} violations"
                    if event.time:
                        text += f" time={event.time}"
                    print(
                        f"{validator.severity}! {column} {message},{text} args={all_args}"
                    )
        return event


class MapValues(StepToDict, MLRunStep):
    """Map column values to new values"""

    def __init__(
        self,
        mapping: Dict[str, Dict[str, Any]],
        with_original_features: bool = False,
        suffix: str = "mapped",
        **kwargs,
    ):
        """Map column values to new values

        example::

            # replace the value "U" with '0' in the age column
            graph.to(MapValues(mapping={'age': {'U': '0'}}, with_original_features=True))

            # replace integers, example
            graph.to(MapValues(mapping={'not': {0: 1, 1: 0}}))

            # replace by range, use -inf and inf for extended range
            graph.to(MapValues(mapping={'numbers': {'ranges': {'negative': [-inf, 0], 'positive': [0, inf]}}}))

        :param mapping: a dict with entry per column and the associated old/new values map
        :param with_original_features: set to True to keep the original features
        :param suffix: the suffix added to the column name <column>_<suffix> (default is "mapped")
        :param kwargs: optional kwargs (for storey)
        """
        super().__init__(**kwargs)
        self.mapping = mapping
        self.with_original_features = with_original_features
        self.suffix = suffix

    def _map_value(self, feature: str, value):
        feature_map = self.mapping.get(feature, {})

        # Is this a range replacement?
        if "ranges" in feature_map:
            for val, val_range in feature_map.get("ranges", {}).items():
                min_val = val_range[0] if val_range[0] != "-inf" else -np.inf
                max_val = val_range[1] if val_range[1] != "inf" else np.inf
                if value >= min_val and value < max_val:
                    return val

        # Is it a regular replacement
        return feature_map.get(value, value)

    def _get_feature_name(self, feature) -> str:
        return f"{feature}_{self.suffix}" if self.with_original_features else feature

    def _do_storey(self, event):
        mapped_values = {
            self._get_feature_name(feature): self._map_value(feature, val)
            for feature, val in event.items()
            if feature in self.mapping
        }

        if self.with_original_features:
            mapped_values.update(event)

        return mapped_values

    def _do_pandas(self, event):
        df = pd.DataFrame(index=event.index)
        for feature in event.columns:
            feature_map = self.mapping.get(feature, {})
            if "ranges" in feature_map:
                # create and apply range map
                for val, val_range in feature_map.get("ranges", {}).items():
                    min_val = val_range[0] if val_range[0] != "-inf" else -np.inf
                    max_val = val_range[1] if val_range[1] != "inf" else np.inf
                    feature_map["ranges"][val] = [min_val, max_val]

                matchdf = pd.DataFrame.from_dict(
                    feature_map["ranges"], "index"
                ).reset_index()
                matchdf.index = pd.IntervalIndex.from_arrays(
                    left=matchdf[0], right=matchdf[1], closed="both"
                )
                df[self._get_feature_name(feature)] = matchdf.loc[event[feature]][
                    "index"
                ].values
            elif feature_map:
                # create and apply simple map
                df[self._get_feature_name(feature)] = event[feature].map(
                    lambda x: feature_map[x]
                )

        if self.with_original_features:
            df = pd.concat([event, df], axis=1)
        return df


class Imputer(StepToDict, MLRunStep):
    def __init__(
        self,
        method: str = "avg",
        default_value=None,
        mapping: Dict[str, Any] = None,
        **kwargs,
    ):
        """Replace None values with default values

        :param method:        for future use
        :param default_value: default value if not specified per column
        :param mapping:       a dict of per column default value
        :param kwargs:        optional kwargs (for storey)
        """
        super().__init__(**kwargs)
        self.mapping = mapping
        self.method = method
        self.default_value = default_value

    def _impute(self, feature: str, value):
        if value is None:
            return self.mapping.get(feature, self.default_value)
        return value

    def _do_storey(self, event):
        imputed_values = {
            feature: self._impute(feature, val) for feature, val in event.items()
        }
        return imputed_values

    def _do_pandas(self, event):
        for feature in event.columns:
            val = self.mapping.get(feature, self.default_value)
            if val is not None:
                event[feature].fillna(val, inplace=True)
        return event


class OneHotEncoder(StepToDict, MLRunStep):
    def __init__(self, mapping: Dict[str, List[Union[int, str]]], **kwargs):
        """Create new binary fields, one per category (one hot encoded)

        example::

            mapping = {'category': ['food', 'health', 'transportation'],
                       'gender': ['male', 'female']}
            graph.to(OneHotEncoder(mapping=one_hot_encoder_mapping))

        :param mapping: a dict of per column categories (to map to binary fields)
        :param kwargs:  optional kwargs (for storey)
        """
        super().__init__(**kwargs)
        self.mapping = mapping
        for values in mapping.values():
            for val in values:
                if not (isinstance(val, str) or isinstance(val, (int, np.integer))):
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        "For OneHotEncoder you must provide int or string mapping list"
                    )

    def _encode(self, feature: str, value):
        encoding = self.mapping.get(feature, [])

        if encoding:

            one_hot_encoding = {
                f"{feature}_{OneHotEncoder._sanitized_category(category)}": 0
                for category in encoding
            }
            if value in encoding:
                one_hot_encoding[
                    f"{feature}_{OneHotEncoder._sanitized_category(value)}"
                ] = 1
            else:
                print(f"Warning, {value} is not a known value by the encoding")
            return one_hot_encoding

        return {feature: value}

    def _do_storey(self, event):
        encoded_values = {}

        for feature, val in event.items():
            encoded_values.update(self._encode(feature, val))
        return encoded_values

    def _do_pandas(self, event):

        for key, values in self.mapping.items():
            event[key] = pd.Categorical(event[key], categories=list(values))
            encoded = pd.get_dummies(event[key], prefix=key, dtype=np.int64)
            event = pd.concat([event.loc[:, :key], encoded, event.loc[:, key:]], axis=1)
        event.drop(columns=list(self.mapping.keys()), inplace=True)
        return event

    @staticmethod
    def _sanitized_category(category):
        # replace(" " and "-") -> "_"
        if isinstance(category, str):
            return re.sub("[ -]", "_", category)
        return category


class DateExtractor(StepToDict, MLRunStep):
    """Date Extractor allows you to extract a date-time component"""

    def __init__(
        self,
        parts: Union[Dict[str, str], List[str]],
        timestamp_col: str = None,
        **kwargs,
    ):
        """Date Extractor extract a date-time component into new columns

        The extracted date part will appear as `<timestamp_col>_<date_part>` feature.

        Supports part values:

        * asm8:              Return numpy datetime64 format in nanoseconds.
        * day_of_week:       Return day of the week.
        * day_of_year:       Return the day of the year.
        * dayofweek:         Return day of the week.
        * dayofyear:         Return the day of the year.
        * days_in_month:     Return the number of days in the month.
        * daysinmonth:       Return the number of days in the month.
        * freqstr:           Return the total number of days in the month.
        * is_leap_year:      Return True if year is a leap year.
        * is_month_end:      Return True if date is last day of month.
        * is_month_start:    Return True if date is first day of month.
        * is_quarter_end:    Return True if date is last day of the quarter.
        * is_quarter_start:  Return True if date is first day of the quarter.
        * is_year_end:       Return True if date is last day of the year.
        * is_year_start:     Return True if date is first day of the year.
        * quarter:           Return the quarter of the year.
        * tz:                Alias for tzinfo.
        * week:              Return the week number of the year.
        * weekofyear:        Return the week number of the year.

        example::

            # (taken from the fraud-detection end-to-end feature store demo)
            # Define the Transactions FeatureSet
            transaction_set = fs.FeatureSet("transactions",
                                            entities=[fs.Entity("source")],
                                            timestamp_key='timestamp',
                                            description="transactions feature set")

            # Get FeatureSet computation graph
            transaction_graph = transaction_set.graph

            # Add the custom `DateExtractor` step
            # to the computation graph
            transaction_graph.to(
                    class_name='DateExtractor',
                    name='Extract Dates',
                    parts = ['hour', 'day_of_week'],
                    timestamp_col = 'timestamp',
                )

        :param parts: list of pandas style date-time parts you want to extract.
        :param timestamp_col: The name of the column containing the timestamps to extract from,
                              by default "timestamp"
        """
        super().__init__(**kwargs)
        self.timestamp_col = timestamp_col
        self.parts = parts

    def _get_key_name(self, part: str, timestamp_col: str):
        timestamp_col = timestamp_col if timestamp_col else "timestamp"
        return f"{timestamp_col}_{part}"

    def _extract_timestamp(self, event):
        # Extract timestamp
        if self.timestamp_col is None:
            timestamp = event["timestamp"]
        else:
            try:
                timestamp = event[self.timestamp_col]
            except KeyError:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"{self.timestamp_col} does not exist in the event"
                )
        return timestamp

    def _do_storey(self, event):
        timestamp = self._extract_timestamp(event)
        # Extract specified parts
        timestamp = pd.Timestamp(timestamp)
        for part in self.parts:
            # Extract part
            extracted_part = getattr(timestamp, part)
            # Add to event
            event[self._get_key_name(part, self.timestamp_col)] = extracted_part
        return event

    def _do_pandas(self, event):
        timestamp = self._extract_timestamp(event)
        # Extract specified parts
        for part in self.parts:
            # Extract part and add it to event
            event[self._get_key_name(part, self.timestamp_col)] = timestamp.map(
                lambda x: getattr(pd.Timestamp(x), part)
            )
        return event


class SetEventMetadata(MapClass):
    """Set the event metadata (id, key, timestamp) from the event body"""

    def __init__(
        self,
        id_path: str = None,
        key_path: str = None,
        time_path: str = None,
        random_id: bool = None,
        **kwargs,
    ):
        """Set the event metadata (id, key, timestamp) from the event body

        set the event metadata fields (id, key, and time) from the event body data structure
        the xx_path attribute defines the key or path to the value in the body dict, "." in the path string
        indicate the value is in a nested dict e.g. `"x.y"` means `{"x": {"y": value}}`

        example::

            flow = function.set_topology("flow")
            # build a graph and use the SetEventMetadata step to extract the id, key and path from the event body
            # ("myid", "mykey" and "mytime" fields), the metadata will be used for following data processing steps
            # (e.g. feature store ops, time/key aggregations, write to databases/streams, etc.)
            flow.to(SetEventMetadata(id_path="myid", key_path="mykey", time_path="mytime"))
                .to(...)  # additional steps

            server = function.to_mock_server()
            event = {"myid": "34", "mykey": "123", "mytime": "2022-01-18 15:01"}
            resp = server.test(body=event)

        :param id_path:   path to the id value
        :param key_path:  path to the key value
        :param time_path: path to the time value (value should be of type str or datetime)
        :param random_id: if True will set the event.id to a random value
        """
        kwargs["full_event"] = True
        super().__init__(**kwargs)
        self.id_path = id_path
        self.key_path = key_path
        self.time_path = time_path
        self.random_id = random_id

        self._tagging_funcs = []

    def post_init(self, mode="sync"):
        def add_metadata(name, path, operator=str):
            def _add_meta(event):
                value = get_in(event.body, path)
                setattr(event, name, operator(value))

            return _add_meta

        def set_random_id(event):
            event.id = uuid.uuid4().hex

        self._tagging_funcs = []
        if self.id_path:
            self._tagging_funcs.append(add_metadata("id", self.id_path))
        if self.key_path:
            self._tagging_funcs.append(add_metadata("key", self.key_path))
        if self.time_path:
            self._tagging_funcs.append(
                add_metadata("time", self.time_path, get_event_time)
            )
        if self.random_id:
            self._tagging_funcs.append(set_random_id)

    def do(self, event):
        for func in self._tagging_funcs:
            func(event)
        return event


class DropFeatures(StepToDict, MLRunStep):
    def __init__(self, features: List[str], **kwargs):
        """Drop all the features from feature list

        :param features:    string list of the features names to drop

        example::

            feature_set = fs.FeatureSet("fs-new",
                                        entities=[fs.Entity("id")],
                                        description="feature set",
                                        engine="pandas",
                                        )
            # Pre-processing grpah steps
            feature_set.graph.to(DropFeatures(features=["age"]))
            df_pandas = fs.ingest(feature_set, data)

        """
        super().__init__(**kwargs)
        self.features = features

    def _do_storey(self, event):
        for feature in self.features:
            try:
                del event[feature]
            except KeyError:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"The ingesting data doesn't contain a feature named '{feature}'"
                )
        return event

    def _do_pandas(self, event):
        return event.drop(columns=self.features)
