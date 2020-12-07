from storey import MapClass


class ValidatorStep(MapClass):
    def __init__(self, featureset=None, **kwargs):
        super().__init__(full_event=True, **kwargs)
        self._validators = {}
        if not self.context:
            return
        self.featureset = self.context.get_feature_set(featureset)
        for key, feature in self.featureset.spec.features.items():
            if feature.validator:
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
