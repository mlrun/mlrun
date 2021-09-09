from typing import Any, Dict, List

import pandas as pd
from storey import MapClass

this_path = "mlrun.feature_store.steps"


class FeaturesetValidator(MapClass):
    def __init__(self, featureset=None, columns=None, name=None, **kwargs):
        super().__init__(full_event=True, **kwargs)
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

    def to_dict(self):
        return {
            "class_name": this_path + ".FeaturesetValidator",
            "name": self.name or "FeaturesetValidator",
            "class_args": {"featureset": self.featureset, "columns": self.columns},
        }


class MapValues(MapClass):
    def __init__(
        self,
        mapping: Dict[str, Dict[str, Any]],
        with_original_features: bool = False,
        suffix: str = "mapped",
        **kwargs,
    ):
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

    def to_dict(self):
        return {
            "class_name": this_path + ".MapValues",
            "name": self.name or "MapValues",
            "class_args": {
                "mapping": self.mapping,
                "with_original_features": self.with_original_features,
                "suffix": self.suffix,
            },
        }


class Imputer(MapClass):
    def __init__(
        self,
        method: str = "avg",
        default_value=None,
        mapping: Dict[str, Dict[str, Any]] = None,
        **kwargs,
    ):
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

    def to_dict(self):
        return {
            "class_name": this_path + ".Imputer",
            "name": self.name or "Imputer",
            "class_args": {
                "mapping": self.mapping,
                "method": self.method,
                "default_value": self.default_value,
            },
        }


class OneHotEncoder(MapClass):
    def __init__(self, mapping: Dict[str, Dict[str, Any]], **kwargs):
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

    def to_dict(self):
        return {
            "class_name": this_path + ".OneHotEncoder",
            "name": self.name or "OneHotEncoder",
            "class_args": {"mapping": self.mapping},
        }


class DateExtractor(MapClass):
    """Date Extractor allows you to extract a date-time component
        from a timestamp feature to a new feature.

        The extracted date part will appear as `<timestamp_col>_<date_part>` feature.

        Parameters
        ----------
        parts : Union[Dict[str, str], List[str]]
            The pandas style date-time parts you want to extract.

            Supports:
            asm8                    Return numpy datetime64 format in nanoseconds.
            day_of_week             Return day of the week.
            day_of_year             Return the day of the year.
            dayofweek               Return day of the week.
            dayofyear               Return the day of the year.
            days_in_month           Return the number of days in the month.
            daysinmonth             Return the number of days in the month.
            freqstr                 Return the total number of days in the month.
            is_leap_year            Return True if year is a leap year.
            is_month_end            Return True if date is last day of month.
            is_month_start          Return True if date is first day of month.
            is_quarter_end          Return True if date is last day of the quarter.
            is_quarter_start        Return True if date is first day of the quarter.
            is_year_end             Return True if date is last day of the year.
            is_year_start           Return True if date is first day of the year.
            quarter                 Return the quarter of the year.
            tz                      Alias for tzinfo.
            week                    Return the week number of the year.
            weekofyear              Return the week number of the year.

        timestamp_col : str, optional
            The name of the column containing the timestamps to extract from,
            by default "timestamp"

        Examples
        --------
        (taken from the fraud-detection end-to-end feature store demo)
        ```
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
        ```
        """

    def __init__(
        self, parts: List[str], timestamp_col: str = None, **kwargs,
    ):
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

    def to_dict(self):
        return {
            "class_name": this_path + ".DateExtractor",
            "name": self.name or "DateExtractor",
            "class_args": {"parts": self.parts, "timestamp_col": self.timestamp_col},
        }
