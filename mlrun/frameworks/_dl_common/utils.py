from abc import ABC
from typing import TypeVar

from .._common import CommonTypes, CommonUtils


class DLTypes(CommonTypes, ABC):
    """
    Deep learning frameworks type hints.
    """

    # A generic type variable for the different tensor type objects of the supported frameworks:
    WeightType = TypeVar("WeightType")


class DLUtils(CommonUtils, ABC):
    """
    Deep learning frameworks utilities.
    """

    pass
