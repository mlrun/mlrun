import math

from mlrun.data_types import ValueType
from mlrun.features import Feature, Validator
from tests.system.base import TestMLRunSystem


# Without enterprise functions (without v3io mount and pipelines)
@TestMLRunSystem.skip_test_if_env_not_configured
class TestFeatureValidator(TestMLRunSystem):
    def validator_types(self):
        for validator in validators_check_type.keys():
            for test_state, test_value in validators_check_type[validator]:
                feature = Feature(validator)
                feature.validator = Validator(True, "Info")

                feature.validator.set_feature(feature)
                ok, result = feature.validator.check(test_value)
                assert ok == test_state, f"Check type error, result:{result}"


# test set for type checks
validators_check_type = {
    # ValueType: [(expected result, testing value), ...]
    ValueType.BOOL: [(True, True), (True, False), (True, "test"), (True, 60)],
    ValueType.INT8: [(True, -128), (True, 127), (False, -129), (False, 128)],
    ValueType.INT16: [(True, -32768), (True, 32767), (False, -32769), (False, 32768)],
    ValueType.INT32: [
        (True, -2147483648),
        (True, 2147483647),
        (False, -2147483649),
        (False, 2147483648),
    ],
    ValueType.INT64: [
        (True, -9223372036854775808),
        (True, 9223372036854775807),
        (False, -9223372036854775809),
        (False, 9223372036854775808),
    ],
    ValueType.INT128: [
        (True, -math.pow(2, 127)),
        (True, math.pow(2, 127) - 1),
        (False, -math.pow(2, 128)),
        (False, math.pow(2, 128)),
    ],
    ValueType.UINT8: [(True, 0), (True, 255), (False, 256), (False, -1)],
    ValueType.UINT16: [(True, 0), (True, 65535), (False, -1), (False, 65536)],
    ValueType.UINT32: [(True, 0), (True, 4294967295), (False, -1), (False, 4294967296)],
    ValueType.UINT64: [
        (True, 0),
        (True, 18446744073709551615),
        (False, -1),
        (False, 18446744073709551616),
    ],
    ValueType.UINT128: [
        (True, 0),
        (True, math.pow(2, 128)),
        (False, -1),
        (False, math.pow(2, 129)),
    ],
    ValueType.FLOAT: [
        (True, 3.1415),
        (True, -273.15),
        (False, "WDC"),
        (True, "6378"),
        (False, ""),
    ],
    ValueType.DOUBLE: [
        (True, 11.2),
        (True, -459.67),
        (False, "WDC"),
        (True, "6378"),
        (False, ""),
    ],
}
