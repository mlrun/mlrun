import mlrun
from mlrun.serving.v2_serving import V2ModelServer
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any

class LLMModelServer(V2ModelServer):

    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        name: str = None,
        model_path: str = None,
        model_name: str = None,
        **kwargs
    ):
        super().__init__(name=name, context=context, model_path=model_path, **kwargs)
        self.model_name = model_name
    
    def load(
        self,
    ):
        # Load the model from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)


    def predict(self, request: dict[str, Any]):
        inputs = request.get("inputs", [])
      
        input_ids, attention_mask = self.tokenizer(
            inputs[0], return_tensors="pt"
        ).values()

        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)

        # Remove input:
        outputs = self.tokenizer.decode(outputs[0])
        outputs = outputs.split(inputs[0])[-1].replace(self.tokenizer.eos_token, "")
        return [{"generated_text": outputs}]
