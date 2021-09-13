import onnx
import onnxruntime

import mlrun


class ONNXMLRunInterface:
    """
    An interface for enabling convenient MLRun features for the PyTorch framework, including training, evaluating and
    automatic logging.
    """

    # MLRun's context default name:
    DEFAULT_CONTEXT_NAME = "mlrun-onnx"

    def __init__(self, model: onnx.ModelProto, context: mlrun.MLClientCtx = None):
        self._model = model
        self._context = context

    def evaluate(self):
        pass

    def predict(self):
        pass
