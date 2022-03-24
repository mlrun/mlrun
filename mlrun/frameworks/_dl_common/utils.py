from typing import TypeVar
from abc import ABC
from .._common import Types, Utils


class DLTypes(Types, ABC):
    """
    Deep learning frameworks type hints.
    """
    # A generic type variable for the different tensor type objects of the supported frameworks:
    WeightType = TypeVar("WeightType")


class DLUtils(Utils, ABC):
    """
    Deep learning frameworks utilities.
    """
    pass
