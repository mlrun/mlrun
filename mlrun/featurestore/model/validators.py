from mlrun.model import ModelObj


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
