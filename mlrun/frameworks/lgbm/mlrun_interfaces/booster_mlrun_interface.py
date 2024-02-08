# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from abc import ABC

import lightgbm as lgb

from ..._common import MLRunInterface
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
        super().add_interface(obj=obj, restoration=restoration)
