from abc import ABC

import lightgbm as lgb

from ..._common import MLRunInterface
from ..._ml_common import MLModelHandler
from ..utils import LGBMTypes


class LGBMBoosterMLRunInterface(MLRunInterface, ABC):
    """
    Interface for adding MLRun features for LightGBM models (Booster API).
    """

    _PROPERTIES = {
        "model_handler": None,  # type: MLModelHandler
    }

    @classmethod
    def add_interface(
        cls,
        obj: lgb.Booster,
        restoration: LGBMTypes.MLRunInterfaceRestorationType = None,
    ):
        """
        Enrich the object with this interface properties, methods and functions, so it will have this LightGBM MLRun's
        features.

        :param obj:         The object to enrich his interface.
        :param restoration: Restoration information tuple as returned from 'remove_interface' in order to add the
                            interface in a certain state.
        """
        super(LGBMBoosterMLRunInterface, cls).add_interface(
            obj=obj, restoration=restoration
        )
