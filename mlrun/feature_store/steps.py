from typing import Any, Dict, List, Union

import pandas as pd
from storey import MapClass

from mlrun.serving.utils import StepToDict


class FeaturesetValidator(StepToDict, MapClass):
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

    def do(self, event):
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


class MapValues(StepToDict, MapClass):
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

        # Is it a string replacement?
        if type(value) is str:
            return feature_map.get(value, value)

        # Is it a range replacement?
        for feature_range in feature_map.get("ranges", []):
            current_range = feature_range["range"]
            if value >= current_range[0] and value < current_range[1]:
                return feature_range["value"]

        # No replacement was made
        return value

    def _feature_name(self, feature) -> str:
        return f"{feature}_{self.suffix}" if self.with_original_features else feature

    def do(self, event):
        mapped_values = {
            self._feature_name(feature): self._map_value(feature, val)
            for feature, val in event.items()
            if feature in self.mapping
        }

        if self.with_original_features:
            mapped_values.update(event)

        return mapped_values


class Imputer(StepToDict, MapClass):
    def __init__(
        self,
        method: str = "avg",
        default_value=None,
        mapping: Dict[str, Dict[str, Any]] = None,
        **kwargs,
    ):
        """Replace None values with default values

        :param method:        for future use
        :param default_value: default value if not specified per column
        :param mapping:       a dict of per column deffault value
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

    def do(self, event):
        imputed_values = {
            feature: self._impute(feature, val) for feature, val in event.items()
        }
        return imputed_values


class OneHotEncoder(StepToDict, MapClass):
    def __init__(self, mapping: Dict[str, Dict[str, Any]], **kwargs):
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

    def _encode(self, feature: str, value):
        encoding = self.mapping.get(feature, [])

        if encoding:
            one_hot_encoding = {f"{feature}_{category}": 0 for category in encoding}
            if value in encoding:
                one_hot_encoding[f"{feature}_{value}"] = 1
            else:
                print(f"Warning, {value} is not a known value by the encoding")
            return one_hot_encoding

        return {feature: value}

    def do(self, event):
        encoded_values = {}
        for feature, val in event.items():
            encoded_values.update(self._encode(feature, val))
        return encoded_values


class DateExtractor(StepToDict, MapClass):
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
            transaction_graph\
                .to(
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

    def do(self, event):
        # Extract timestamp
        if self.timestamp_col is None:
            timestamp = event["timestamp"]
        else:
            try:
                timestamp = event[self.timestamp_col]
            except Exception:
                raise ValueError(f"{self.timestamp_col} does not exist in the event")

        # Extract specified parts
        timestamp = pd.Timestamp(timestamp)
        for part in self.parts:
            # Extract part
            extracted_part = getattr(timestamp, part)
            # Add to event
            event[self._get_key_name(part, self.timestamp_col)] = extracted_part
        return event
