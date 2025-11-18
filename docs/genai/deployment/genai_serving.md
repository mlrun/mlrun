(genai-serving)=
# Serving gen AI models

With MLRun you can serve any model, locally hosted (including pretrained models that are downloaded from the Hugging Face model hub, as well as models that are fine-tuned with MLRun) and remote models. (See [Hugging Face model hub](https://huggingface.co/docs/hub/en/models-the-hub).)
The main differences between serving a gen AI model and any other model are the inputs and outputs: inputs in gen AI are usually unstructured (text or images), and the model is usually a transformer model. 

Another common use case is to serve the model as part of an inference pipeline, where the model is used as part of a larger pipeline that includes data preprocessing, model execution, and post-processing. This is covered in the {ref}`gen AI serving graph section <genai-serving-graph>`.

**In this section**
- [Serving a local model from the MLRun hub](#serving-a-local-model-from-the-mlrun-hub)
- [Serving using a remote model](#serving-using-a-remote-model)
- [Implementing your own model serving function](#implementing-your-own-model-serving-function)

## Serving a local model from the MLRun hub

The hub has a serving class called [`hugging_face_serving`](https://www.mlrun.org/hub/functions/master/hugging_face_serving/) to run Hugging Face models. The following code shows how to import the function to your project:

```python
hugging_face_serving = project.set_function("hub://hugging_face_serving")
```

Next, add the model to the function using this code:

```python
hugging_face_serving.add_model(
    "mymodel",
    class_name="HuggingFaceModelServer",
    model_path="123",  # This is not used, just for enabling the process.
    task="text-generation",
    model_class="AutoModelForCausalLM",
    model_name="openai-community/gpt2",
    tokenizer_class="AutoTokenizer",
    tokenizer_name="openai-community/gpt2",
)
```
### Testing the local model

```python
hugging_face_mock_server = hugging_face_serving.to_mock_server()
result = hugging_face_mock_server.test(
    "/v2/models/mymodel", body={"inputs": ["write a short poem"]}
)
print(f"Output: {result['outputs']}")
```

## Serving using a remote model

There are two types of remote models. 

- When using OpenAI models, you can send requests through the OpenAI API to perform different tasks such as text generation, embeddings, and more. For text generation usage, see the example in {ref}`deploy-openai-model`.
- Hugging Face use Pipeline as a client; it downloads the model and loads it to the RAM.
  - Hugging Face models might required more resources than usual.
  - By default, in LLModel usage, metrics are calculated after invocation. These token metrics are estimates and may not be fully accurate.
  - Hugging Face's Inference Provider is designed to handle OpenAI-style chat format (role/content) and therefore requires models that support `tokenizer.apply_chat_template`. If a model does not provide this functionality, you must implement a manual solution.

### Serving the remote model
The following code shows the basics of serving and deploying a remote model.

```python
graph = function.set_topology("flow", engine="async")
model_runner_step = ModelRunnerStep(name="my_model_runner")
model_runner_step.add_model(
    model_class="LLModel",
    endpoint_name="my_endpoint",
    execution_mechanism=execution_mechanism,
    model_artifact=llm_prompt_artifact,
    result_path="output",
)
graph.to(model_runner_step).respond()

print("Serving graph configured with dedicated_process execution mechanism")

# Deploy the function
print("Deploying function...")
function.deploy()
print("Function deployed successfully!")
```

### Testing the remote model
```python
# Test the model with the input data
response = function.invoke(
    f"v2/models/{mlrun_model_name}/infer",
    json.dumps(INPUT_DATA),
)["output"]

print("Response received:")
print(f"Response length: {len(response)}")
print("\nResponse structure:")
for key in response.keys():
    print(f"  - {key}")
```
## Implementing your own model serving function

The following code shows how to build a simple model serving function using MLRun. The function loads a pretrained model from the Hugging Face model hub and serves it using the MLRun model server.

```{admonition} Note

This example uses the [ONNX runtime](https://onnxruntime.ai/docs/) but it's here for illustrative purposes. You can use any other runtime within your model serving class.

To run this code, make sure to run `pip install huggingface_hub onnxruntime_genai` in your python environment.
```


```python
import os
from typing import Any, Dict

from huggingface_hub import snapshot_download
import onnxruntime_genai as og
import mlrun


class OnnxGenaiModelServer(mlrun.serving.v2_serving.V2ModelServer):
    def __init__(
        self,
        context: mlrun.MLClientCtx,
        name: str,
        model_path: str,
        model_name: str,
        search_options: Dict = {},
        chat_template: str = "<|user|>\n{prompt} <|end|>\n<|assistant|>",
        **class_args,
    ):
        # Initialize the base server:
        super(OnnxGenaiModelServer, self).__init__(
            context=context,
            name=name,
            model_path=model_path,
            **class_args,
        )

        self.chat_template = chat_template
        self.search_options = search_options

        # Set the max length to something sensible by default, unless it is specified by the user,
        # since otherwise it will be set to the entire context length
        if "max_length" not in self.search_options:
            self.search_options["max_length"] = 2048

        # Save hub loading parameters:
        self.model_name = model_name

        # Prepare variables for future use:
        self.model_folder = None
        self.model = None
        self.tokenizer = None

    def load(self):
        # Download the model snapshot and save it to the model folder
        self.model_folder = snapshot_download(self.model_name)

        # Load the model from the model folder
        self.model = og.Model(os.path.join(self.model_folder, self.model_path))

        # Create a tokenizer using the loaded model
        self.tokenizer = og.Tokenizer(self.model)

    def predict(self, request: Dict[str, Any]) -> list:
        # Get prompts from inputs::
        prompts = [
            f'{self.chat_template.format(prompt=input.get("prompt"))}'
            for input in request["inputs"]
        ]

        # Tokenize:
        input_tokens = self.tokenizer.encode_batch(prompts)

        # Create the parameters
        params = og.GeneratorParams(self.model)
        params.set_search_options(**self.search_options)
        params.input_ids = input_tokens

        # Generate output tokens:
        output_tokens = self.model.generate(params)

        # Decode output tokens to text:
        response = [
            {"prediction": self.tokenizer.decode(output), "prompt": prompt}
            for (output, prompt) in zip(output_tokens, prompts)
        ]

        return response
```

During load, the code above downloads a model from the Hugging Face hub and creates a model object and a tokenizer.

During prediction, the code collects all prompts, tokenizes the prompts, generates the response tokens, and decodes the output tokens to text.

Save the code above to `src/onnx_genai_serving.py` and then create a model serving function with the following code:

``` python
import os
import mlrun

project = mlrun.get_or_create_project(
    "genai-deployment", context="./", user_project=True
)

genai_serving = project.set_function(
    "src/onnx_genai_serving.py",
    name="genai-serving",
    kind="serving",
    image="mlrun/mlrun",
    requirements=["huggingface_hub", "onnxruntime_genai"],
)

genai_serving.add_model(
    "mymodel",
    model_name="microsoft/Phi-3-mini-4k-instruct-onnx",
    model_path=os.path.join("cpu_and_mobile", "cpu-int4-rtn-block-32-acc-level-4"),
    class_name="OnnxGenaiModelServer",
)
```

The code loads a Phi-3 model. This example uses the CPU version so it's easy to test and run, but you can just as easily provide a GPU-based model.

Test the model with the following code:

```python
mock_server = genai_serving.to_mock_server()

result = mock_server.test(
    "/v2/models/mymodel", body={"inputs": [{"prompt": "What is 1+1?"}]}
)
print(f"Output: {result['outputs']}")
```

A typical output would be:
```
Output: [{'prediction': '\nWhat is 1+1? \n1+1 equals 2. This is a basic arithmetic addition problem where you add one unit to another unit.', 'prompt': '<|user|>\nWhat is 1+1? <|end|>\n<|assistant|>'}]
```

To deploy the model, run:
```python
project.deploy_function(genai_serving)
```

This builds a docker images with the required dependencies and deploys a Nuclio function.

To test the model, use the HTTP trigger:
```python
genai_serving.invoke(
    "/v2/models/mymodel", body={"inputs": [{"prompt": "What is 1+1?"}]}
)
```
