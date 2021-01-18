from typing import Dict, Any
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
