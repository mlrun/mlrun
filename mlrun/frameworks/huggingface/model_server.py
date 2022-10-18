# Copyright 2018 Iguazio
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

import json
from importlib import import_module
from typing import Any, Dict, List

from transformers import pipeline

import mlrun
from mlrun.serving.v2_serving import V2ModelServer

PACKAGE_MODULE = "transformers"


def json_serializable(obj):
    try:
        json.dumps(obj=obj)
        return True
    except (TypeError, OverflowError):
        return False


class HuggingFaceModelServer(V2ModelServer):
    """
    Hugging Face Model serving class, inheriting the V2ModelServer class for being initialized automatically by the
    model server and be able to run locally as part of a nuclio serverless function, or as part of a real-time pipeline.

    Notice:
        In order to use this serving class, please ensure that the transformers package is installed.
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx,
        name: str,
        task: str,
        model_path: str = None,
        model_name: str = None,
        model_class: str = None,
        tokenizer_name: str = None,
        tokenizer_class: str = None,
        framework: str = None,
        **class_args,
    ):
        """
        Initialize a serving class for a Hugging face model.

        :param context:         The mlrun context to work with
        :param name:            The name of this server to be initialized
        :param model_path:      Not in use. When adding a model pass any string value
        :param model_name:      The model's name in the Hugging Face hub
                                e.g., `nlptown/bert-base-multilingual-uncased-sentiment`
        :param model_class:     The model class type object which can be passed as the class's name (string).
                                Must be provided and to be matched with `model_name`.
                                e.g., `AutoModelForSequenceClassification`
        :param tokenizer_name:  The name of the tokenizer in the Hugging Face hub
                                e.g., `nlptown/bert-base-multilingual-uncased-sentiment`
        :param tokenizer_class: The model's class type object which can be passed as the class's name (string).
                                Must be provided and to be matched with `model_name`.
                                e.g., `AutoTokenizer`
        :param framework:       The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified
                                framework must be installed.
                                If no framework is specified, will default to the one currently installed.
                                If no framework is specified and both frameworks are installed, will default to the
                                framework of the `model`, or to PyTorch if no model is provided
        :param class_args:      -
        """
        super(HuggingFaceModelServer, self).__init__(
            context=context,
            name=name,
            model_path=model_path,
            **class_args,
        )
        self.task = task
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        self.framework = framework
        self.pipe = None

    def load(self):
        """
        loads the model and the tokenizer and initializes the pipeline based on them.
        """

        # Loading the pretrained model:
        if self.model_class:
            model_object = getattr(import_module(PACKAGE_MODULE), self.model_class)
            self.model = model_object.from_pretrained(self.model_name)

        # Loading the pretrained tokenizer:
        if self.tokenizer_class:
            tokenizer_object = getattr(
                import_module(PACKAGE_MODULE), self.tokenizer_class
            )
            self.tokenizer = tokenizer_object.from_pretrained(self.tokenizer_name)

        # Initializing the pipeline:
        self.pipe = pipeline(
            task=self.task,
            model=self.model or self.model_name,
            tokenizer=self.tokenizer,
            framework=self.framework,
        )

    def predict(self, request: Dict[str, Any]) -> List:
        """
        Generate model predictions from sample.

        :param request: The request to the model. The input to the model will be read from the "inputs" key.

        :return: The model's prediction on the given input.
        """
        if self.pipe is None:
            raise ValueError("Please use `.load()`")

        try:
            # Predicting:
            if isinstance(request["inputs"][0], dict):
                result = [self.pipe(**_input) for _input in request["inputs"]]
            else:
                result = self.pipe(request["inputs"])

            # replace list of lists of dicts into a list of dicts:
            # The result may vary from one model to another.
            if all(isinstance(res, list) for res in result):
                result = [res[0] for res in result]

            # Converting JSON non-serializable objects into strings:
            non_serializable_types = []
            for res in result:
                for key, val in res.items():
                    if not json_serializable(val):
                        non_serializable_types.append(str(type(val)))
                        res[key] = str(val)
            if non_serializable_types:
                self.context.logger.info(
                    f"Non-serializable types: {non_serializable_types} were casted to strings"
                )
        except Exception as e:
            raise Exception("Failed to predict %s" % e)
        return result

    def explain(self, request: Dict) -> str:
        """
        Return a string explaining what model is being served in this serving function and the function name.

        :param request: A given request.

        :return: Explanation string.
        """
        return f"The '{self.model_name}' model serving function named '{self.name}"
