import os
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import mlrun
from mlrun.serving.v2_serving import V2ModelServer


class LLMModelServer(V2ModelServer):
    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        name: str = None,
        model_path: str = None,
        llm_type: str = "HuggingFace",
        model_name: str = None,
        **kwargs,
    ):
        """
        Initialize a serving class for general llm usage.
        :param context:         For internal use (passed in init).
        :param name:            The name of this server to be initialized
        :param model_path:      If the model is already trained and saved, the path to the model file.
        :param llm_type:        The type of the llm platform to use. Currently only "OpenAI" is supported.
        :param model_name:      The model's name to use in the llm platform.
        """
        super().__init__(name=name, context=context, model_path=model_path, **kwargs)
        self.llm_type = llm_type
        self.model = None
        self.model_name = model_name

        self.my_kwargs = {}

        adapter = kwargs.pop("adapter", None)
        if adapter:
            self.my_kwargs["adapter"] = adapter

        device_map = kwargs.pop("device_map", None)
        if device_map:
            self.my_kwargs["device_map"] = device_map

        self.generate_kwargs = kwargs.pop("generate_kwargs", {})

    def load(
        self,
    ):
        self.model = PLATFORM_MAPPING[self.llm_type](
            self.context, model_name=self.model_name, **self.my_kwargs
        )

    def predict(self, request: dict[str, Any]):
        inputs = request.get("inputs", [])
        kwargs = request.get("kwargs", self.generate_kwargs)
        return [self.model.invoke(inputs, **kwargs)[0]["generated_text"]]


class PlatformHandler:
    def __init__(self, context, model_name, **kwargs):
        self.context = context
        self.model_name = model_name

    def invoke(self, inputs, **kwargs):
        return self._invoke(inputs, **kwargs)

    def _invoke(self, inputs, **kwargs):
        pass


class HuggingFaceHandler(PlatformHandler):
    def __init__(self, context, model_name, task="text-generation", **kwargs):
        super().__init__(context, model_name)
        # Look for the HuggingFace API token in the environment variables or secrets:
        huggingface_api_token = context.get_secret(key="HF_TOKEN")
        if huggingface_api_token:
            os.environ["HF_TOKEN"] = huggingface_api_token

        self.model_name = model_name
        # Load the HuggingFace model using langchain:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        device_map = kwargs.get("device_map", "auto")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map=device_map,
        )

        # Merge adapter with base model
        if "adapter" in kwargs:
            model = PeftModel.from_pretrained(self.model, kwargs["adapter"])
            self.model = model.merge_and_unload()

    def _invoke(self, inputs, **kwargs):
        input_ids, attention_mask = self.tokenizer(
            inputs[0], return_tensors="pt"
        ).values()
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        # Remove input:
        outputs = self.tokenizer.decode(outputs[0])
        outputs = outputs.split(inputs[0])[-1].replace(self.tokenizer.eos_token, "")
        return [{"generated_text": outputs}]


# Holds names of PlatformHandler classes
class PlatformTypes:
    OPENAI = "OpenAI"
    COHERE = "Cohere"
    HUGGINGFACE = "HuggingFace"
    ANTHROPIC = "Anthropic"


# Maps Platform types to their handlers
PLATFORM_MAPPING = {
    PlatformTypes.HUGGINGFACE: HuggingFaceHandler,
}
