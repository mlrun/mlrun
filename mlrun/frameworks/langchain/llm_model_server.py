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
    * ainvoke (TBD)
    * abatch (TBD)
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
        init_method: str = None,
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
        :param init_method:       The initialization method to use while initializing the llm, default is __init__.
        :param init_kwargs:       The initialization arguments to use while initializing the llm.
        :param generation_kwargs: The generation arguments to use while generating text.
        :param name:              The name of this server to be initialized.
        :param model_path:        Not in use. When adding a model pass any string value
                #TODO support loading a chain as pickle, then path is to pickle file or artifact
        """
        super().__init__(name=name, context=context, model_path=model_path)
        self.llm = llm
        self.init_method = init_method  # None=__init__
        self.init_kwargs = init_kwargs or {}
        self.generation_kwargs = generation_kwargs or {}

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
        model_class = getattr(langchain_community.llms, self.llm)
        if self.init_method:
            self.model = getattr(model_class, self.init_method)(**self.init_kwargs)
        else:
            self.model = model_class(**self.init_kwargs)

    def predict(
        self,
        request: Dict[str, Any],
    ):
        """
        Predict the output of the model, can use the following usages:

        * predict
        * invoke
        * batch
        * ainvoke (TBD)
        * abatch (TBD)

        Upon receiving a request, the model will use the "usage" key to determine the usage of the model, and will
        return the model's prediction accordingly.

        :param request:           The request to the model. The input to the model will be read from the "inputs" key.
        :return:                  The model's prediction on the given input.
        """
        inputs = request.get("inputs", [])
        usage = request.get("usage", "predict")
        generation_kwargs = (
            request.get("generation_kwargs", None) or self.generation_kwargs
        )
        if usage == "predict":
            return self.model.invoke(input=inputs[0], config=generation_kwargs)
        elif usage == "invoke":
            config = request.get("config", None)
            stop = request.get("stop", None)
            ans = self.model.invoke(
                input=inputs[0], config=config, stop=stop, **generation_kwargs
            )
            return ans
        elif usage == "batch":
            config = request.get("config", None)
            return_exceptions = request.get("return_exceptions", None)
            return self.model.batch(
                inputs=inputs,
                config=config,
                return_exceptions=return_exceptions,
                **generation_kwargs,
            )
        elif usage == "ainvoke":
            raise NotImplementedError("ainvoke is not implemented")
        elif usage == "abatch":
            raise NotImplementedError("abatch is not implemented")
        else:
            raise ValueError(f"Unknown usage: {usage}")
