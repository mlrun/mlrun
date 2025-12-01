(model-serving-steps)=
# Model serving steps
Learn about the ModelRunnerStep and other steps used when serving models.

**In this section**
- [ModelRunnerStep](#modelrunnerstep)
- [Router step](#router-step)

## ModelRunnerStep

The {py:class}`~mlrun.serving.ModelRunnerStep` gives you an advanced way to run multiple models on each event with control 
over how they are executed in terms of concurrency and parallelism. For example, it supports
running models in a multi-process or a multi-threaded paradigm, and it supports having a dedicated process for a given
model (useful when the model has a long startup time or requires a lot of resources). Different execution mechanisms can be
used for different models within the same step. ModelRunnerStep supports a shared
model that is invoked from multiple steps in one graph. Model endpoints resresent the models themselves, not the steps.
See [Basic code examples](#basic-code-examples) and {py:meth}`mlrun.serving.ModelRunnerStep.add_model`.

ModelRunnerSteps have model endpoints, and can therefore be monitored. The input and output of each step are user-configurable. See {py:meth}`mlrun.serving.ModelRunnerStep.add_model`.

When a `ModelRunnerStep `is included in a function graph, MLRun automatically imports the default language model class (`LLModel` or `mlrun.serving.states.LLModel`) during function deployment to wrap the model for handling a LLM prompt-based inference. This class extends the base Model to provide specialized handling for `LLMPromptArtifact` objects, enabling both synchronous and asynchronous invocation of language models. Follow the class description and implement your own enrichment when a custom class is needed.

ModelRunnerStep can only be added to a graph that has the [flow topology](../serving/deploying-graphs.ipynb#flow) and running with the async engine, giving better utilization of CPU/GPU.

### SDK
- {py:meth}`~mlrun.serving.ModelRunnerStep.add_model`: adds a model to the model runner and configures its execution.
- {py:meth}`~mlrun.serving.ModelRunnerStep.add_shared_model_proxy`: Adds a proxy model to the ModelRunnerStep. 
- {py:meth}`~mlrun.serving.ModelSelector`: Select which model to run on each event.

### Preprocess steps

When adding models to the `ModelRunnerStap`, there are many configuration options, for example, excluding unnecessary details that are included in any LLM, input and outputs, which can be paths, dict, etc. 
See the parameters in {py:meth}`~mlrun.serving.ModelRunnerStep.add_model`.

### Basic code examples

This code illustrates a `ModelRunnerStap` with two models. The `ModelSelector` determines which model to run on each event, based on responses from an LLM (for example, finanace vs. travel). It can be a class or a string. If you do not provide a `ModelSelector` to the `ModelRunnerStep `the default case is to run all models.

```
from mlrun.serving import ModelRunnerStep, ModelSelector

class MyClassifier(ModelSelector):
    def __init__(self, models: Union[list[str], list[Model]]):
        super().__init__()
        self.models = deepcopy(models)

    def select(
        self, event, available_models: list[Model]
    ) -> Union[list[str], list[Model]]:
        current_models = event.body.get("models")
        if current_models and set(current_models).issubset(set(self.models)):
            return current_models
        return []

function = project.set_function(
    name="my-project",
    kind="serving",
    tag="latest",
    func="my-func",
    image=image,
    requirements=["openai==1.77.0"],
)
graph = function.set_topology("flow", engine="async")

model_runner_step = ModelRunnerStep(
    name="model_runner_step",
    model_selector="MyClassifier",  # Classify which model should be used
)

model_runner_step.add_model(
    endpoint_name="endpoint-1",
    model_artifact=llm_prompt_artifact-1,
    execution_mechanism="thread_pool",
    model_class="LLModel",
)
model_runner_step.add_model(
    endpoint_name="endpoint-2",
    model_artifact=llm_prompt_artifact-2,
    execution_mechanism="process_pool",
    model_class="LLModel",
)

graph.to(model_runner_step).respond()
```     



### Shared models 

Use the `add_shared_model` method to add a shared model to the graph: this model becomes accessible to all `ModelRunnerSteps` in the graph. Use `add_shared_model_proxy` to add a proxy model to a `ModelRunnerStep`. A proxy model acts as a lightweight reference to an existing shared model within the graph. It allows each step to reuse the same underlying shared model without duplicating it, while still being able to assign a unique endpoint name, labels, and endpoint creation strategy for tracking or monitoring purposes. This helps maintain efficiency and consistency across multiple model runners that operate on shared models. See an example in the tutorial [Using LLM prompt templates and artifacts](../tutorials/genai-04-llm-prompt-artifact.ipynb#define-the-function-graph-and-add-modelrunnerstep-with-proxy-models-for-the-shared-model)

### Output

The `predict()` method automatically returns a result with the following schema:
```
body[result_path] = {
    "answer": ...,  # The model's response
    "usage": {...}  # Token usage metadata
}
```
The usage is different for OpenAI (Chat Completions format) and HuggingFace (Text Generation pipeline).

MLRun extracts and saves the entire usage dictionary from the **OpenAI** response, for full transparency and monitoring. For example:
```
"usage": {
  "completion_tokens": 59,
  "prompt_tokens": 14,
  "total_tokens": 73,
  "completion_tokens_details": {
    "accepted_prediction_tokens": 0,
    "audio_tokens": 0,
    "reasoning_tokens": 0,
    "rejected_prediction_tokens": 0
  },
  "prompt_tokens_details": {
    "audio_tokens": 0,
    "cached_tokens": 0
  }
}
```

**Hugging Face** does not natively return token usage, so MLRun approximates these fields based on the prompt and generated answer:
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`

These estimates are less precise than OpenAI, but they do provide useful visibility. 
Example:
```
"usage": {
  "prompt_tokens": 48,
  "completion_tokens": 101,
  "total_tokens": 149
}
```

## Router step

{py:class}~`mlrun.serving.RouterStep` implements routing logic for running child routes. See the example in {ref}`graph-example`.