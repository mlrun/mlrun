# flake8: noqa  - this is until we take care of the F401 violations with respect to __all__ & sphinx

from .data_types import InferOptions, ValueType, pd_schema_to_value_type
from .infer import DFDataInfer


class BaseDataInfer:
    infer_schema = None
    get_preview = None
    get_stats = None


def get_infer_interface(df) -> BaseDataInfer:
    if hasattr(df, "rdd"):
        from .spark import SparkDataInfer

        return SparkDataInfer
    return DFDataInfer
