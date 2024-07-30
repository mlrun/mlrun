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
# flake8: noqa  - this is until we take care of the F401 violations with respect to __all__ & sphinx

from typing import Any, Dict, Union

import langchain_community.llms
from langchain_core.language_models.llms import LLM

import mlrun
from mlrun.serving.v2_serving import V2ModelServer


class LangChainModelServer(V2ModelServer):
    """
     LangChain Model serving class, inheriting the V2ModelServer class to be able to run locally or as part of a Nuclio
      serverless function.

    For compatibility with langchain models, this class supports the following methods:

    * invoke
    * batch
    * ainvoke
    * abatch
    * stream (TBD)
    * astream (TBD)
    This class can serve a LangChain chain or llm:

    * `LLM` - Expose a `langchain.LLM` object as a serverless function. Can work in offline and online modes:
      * Offline: Get an initialized `LLM` object - only in local testing (mock server)
      * Online: Get a class name and initialization keyword arguments to initialize the `LLM` object and serve it.
    * `Chain` (TBD) - Expose an entire `langchain` object as a pickled model logged to MLRun.
    > Notice: In order to use this serving class, please ensure that the `langchain` and `langchain_community` packages
     are installed.
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        llm: Union[str, LLM] = None,
        init_kwargs: Dict[str, Any] = None,
        generation_kwargs: Dict[str, Any] = None,
        name: str = None,
        model_path: str = None,
        **kwargs,
    ):
        """
        Initialize a serving class for general llm usage.
        :param context:           The mlrun context to use.
        :param llm:               The llm object itself in case of local usage or the name of the llm.
        :param init_kwargs:       The initialization arguments to use while initializing the llm.
        :param generation_kwargs: The generation arguments to use while generating text.
        :param name:              The name of this server to be initialized.
        :param model_path:        Not in use. When adding a model pass any string value
                #TODO support loading a chain as pickle, then path is to pickle file or artifact
        """
        super().__init__(name=name, context=context, model_path=model_path)
        self.llm = llm
        self.init_kwargs = init_kwargs or {}
        self.generation_kwargs = generation_kwargs

    def load(self):
        """
        loads the model.
        """
        # If the llm is already an LLM object, use it directly
        # TODO: Add support for chains
        if isinstance(self.llm, LLM):
            self.model = self.llm
            return
        # If the llm is a string (or not given, then we take default model), load the llm from langchain.
        self.model = getattr(langchain_community.llms, self.llm)(**self.init_kwargs)

    def predict(
        self, request: Dict[str, Any], generation_kwargs: Dict[str, Any] = None
    ):
        """
        Predict the output of the model.
        :param request:           The request to the model. The input to the model will be read from the "inputs" key.
        :param generation_kwargs: The generation arguments to use while generating response.
        :return:                  The model's prediction on the given input.
        """
        inputs = request.get("inputs", [])
        generation_kwargs = generation_kwargs or self.generation_kwargs
        return self.model.invoke(input=inputs[0], config=generation_kwargs)

    def op_invoke(
        self, request: Dict[str, Any], generation_kwargs: Dict[str, Any] = None
    ):
        """
        Invoke the model. (Same as predict)
        :param request:           The request to the model. The input to the model will be read from the "body.inputs"
                                  key.
        :param generation_kwargs: The generation arguments to use while generating response.
        :return:                  The model's prediction on the given input.
        """
        request = request.body
        inputs = request.get("inputs", [])
        generation_kwargs = generation_kwargs or self.generation_kwargs
        return self.model.invoke(input=inputs[0], config=generation_kwargs)

    def op_batch(
        self, request: Dict[str, Any], generation_kwargs: Dict[str, Any] = None
    ):
        """
        Invoke the model in batch.
        :param request:           The request to the model. The input to the model will be read from the "body.inputs"
                                  key.
        :param generation_kwargs: The generation arguments to use while generating response.
        :return:                  The model's prediction on the given input.
        """
        request = request.body
        inputs = request.get("inputs", [])
        generation_kwargs = generation_kwargs or self.generation_kwargs
        return self.model.batch(inputs=inputs, config=generation_kwargs)

    def op_ainvoke(
        self, request: Dict[str, Any], generation_kwargs: Dict[str, Any] = None
    ):
        """
        Invoke the model asynchronously.
        :param request:           The request to the model. The input to the model will be read from the "body.inputs"
                                  key.
        :param generation_kwargs: The generation arguments to use while generating response.
        :return:                  The model's prediction on the given input.
        """
        request = request.body
        inputs = request.get("inputs", [])
        generation_kwargs = generation_kwargs or self.generation_kwargs
        response = self.model.ainvoke(input=inputs, config=generation_kwargs)
        return response

    def op_abatch(
        self, request: Dict[str, Any], generation_kwargs: Dict[str, Any] = None
    ):
        """
        Invoke the model asynchronously in batches.
        :param request:           The request to the model. The input to the model will be read from the "body.inputs"
                                  key.
        :param generation_kwargs: The generation arguments to use while generating response.
        :return:                  The model's prediction on the given input.
        """
        request = request.body
        inputs = request.get("inputs", [])
        generation_kwargs = generation_kwargs or self.generation_kwargs
        response = self.model.abatch(inputs=inputs, config=generation_kwargs)
        return response
