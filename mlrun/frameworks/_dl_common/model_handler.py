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
