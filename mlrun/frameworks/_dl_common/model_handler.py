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

from .._common import ModelHandler


class DLModelHandler(ModelHandler, ABC):
    """
    Abstract class for a deep learning framework model handling, enabling loading, saving and logging it during runs.
    """

    # Constant artifact names:
    _WEIGHTS_FILE_ARTIFACT_NAME = "{}_weights_file"

    def _get_weights_file_artifact_name(self) -> str:
        """
        Get the standard name for the weights file artifact.

        :return: The weights file artifact name.
        """
        return self._WEIGHTS_FILE_ARTIFACT_NAME.format(self._model_name)
