from storey import MapClass

from mlrun.featurestore.model import FeatureAggregation
from mlrun.serving.states import TaskState

this_path = "mlrun.featurestore.steps"


class FeaturesetValidator(MapClass):
    def __init__(self, featureset=None, columns=None, name=None, **kwargs):
        super().__init__(full_event=True, **kwargs)
        self._validators = {}
        self.featureset = featureset or "."
        self.columns = columns
        self.name = name
        if not self.context:
            return
        self._featureset = self.context.get_feature_set(featureset)
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

    def to_state(self):
        return TaskState(
            this_path + ".FeaturesetValidator",
            name=self.name or "FeaturesetValidator",
            class_args={"featureset": self.featureset, "columns": self.columns},
        )
