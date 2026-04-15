(model-serving-steps)=
# Model serving steps
Learn about the ModelRunnerStep and the HTTP streaming step.

**In this section**
- [ModelRunnerStep](#modelrunnerstep)
- [HTTP streaming step](#http-streaming-step)

## ModelRunnerStep

### Description

The {py:class}`~mlrun.serving.states.ModelRunnerStep` gives you an advanced way to run multiple models on each event with control 
over how they are executed in terms of concurrency and parallelism. For example, it supports
running models in a multi-process or a multi-threaded paradigm, and it supports having a dedicated process for a given
model (useful when the model has a long startup time or requires a lot of resources). Different execution mechanisms can be
used for different models within the same step. ModelRunnerStep supports a shared
model that is invoked from multiple steps in one graph. Model endpoints represent the models themselves, not the steps.

ModelRunnerSteps have model endpoints, and can therefore be monitored. The input and output of each step are user-configurable. See [Example with classifier](#example-with-classifier) and {py:meth}`~mlrun.serving.states.ModelRunnerStep.add_model`.

When a `ModelRunnerStep` is included in a graph, MLRun automatically imports the default language model class (`LLModel` or `mlrun.serving.states.LLModel`) during function deployment to wrap the model for handling a LLM prompt-based inference. This class extends the base Model to provide specialized handling for `LLMPromptArtifact` objects, enabling both synchronous and asynchronous invocation of language models. Follow the class description and implement your own enrichment when a custom class is needed.

ModelRunnerStep can only be added to a graph that has the {ref}`flow topology<flow-topology>` and running with the async engine, giving better utilization of CPU/GPU.

ModelRunnerStep is used to execute and manage individual steps within a machine learning model pipeline. Common use cases include:
- Running inference or prediction tasks as part of a larger workflow.
- Orchestrating sequential or parallel model steps, such as data preprocessing, feature extraction, and model evaluation.
- Integrating model steps into automated CI/CD pipelines for machine learning.
- Monitoring and logging the performance and outputs of each step for debugging and optimization.
- Enabling modular and reusable components in ML workflows, allowing teams to update or swap out steps independently.

### SDK
- {py:meth}`~mlrun.serving.states.ModelRunnerStep.add_model`: adds a model to the model runner and configures its execution.
- {py:meth}`~mlrun.serving.states.ModelRunnerStep.add_shared_model_proxy`: Adds a proxy model to the ModelRunnerStep. 
- {py:meth}`~mlrun.serving.ModelSelector`: Select which model to run on each event.

### Preprocess steps

When adding models to the `ModelRunnerStap`, there are many configuration options, for example, excluding unnecessary details that are included in any LLM, input and outputs, which can be paths, dict, etc. 
See the parameters in {py:meth}`~mlrun.serving.states.ModelRunnerStep.add_model`.

### Shared models 

Use the `add_shared_model` method to add a shared model to a graph: this model becomes accessible to all `ModelRunnerSteps` in the graph. Use `add_shared_model_proxy` to add a proxy model to a `ModelRunnerStep`. A proxy model acts as a lightweight reference to an existing shared model within the graph. It allows each step to reuse the same underlying shared model without duplicating it, while still being able to assign a unique endpoint name, labels, and endpoint creation strategy for tracking or monitoring purposes. This helps maintain efficiency and consistency across multiple model runners that operate on shared models. See an example in the tutorial [Using LLM prompt templates and artifacts](../tutorials/genai-04-llm-prompt-artifact.ipynb#define-the-function-graph-and-add-modelrunnerstep-with-proxy-models-for-the-shared-model)

### Example with classifier

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
    model_runner_selector="MyClassifier",  # Classify which model should be used
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

### Example with batching
Example of batching using `ModelRunnerStep`. See a full flow in {ref}`hf-model-batch-serving-graph`.

```
graph = function.set_topology("flow", engine="async")
step = graph.to(
    "storey.Batch",
    "my_batching",
    max_events=2,
    flush_after_seconds=4,
    full_event=True,
)
model_runner_step = ModelRunnerStep(name="my_model_runner")
model_runner_step.add_model(
    model_class="mlrun.serving.states.LLModel",
    endpoint_name="my_endpoint",
    execution_mechanism="dedicated_process",
    model_artifact=llm_prompt_artifact,
    result_path="output",
)
step = step.to(model_runner_step)
step.to("storey.FlatMap", _fn="(event.body)", full_event=True).respond()

print("Serving graph configured with dedicated_process execution mechanism")

#  Enable AsyncSpec when using batch step
function.with_http(async_spec=AsyncSpec())
```

## HTTP streaming step

A streaming step is invoked with a single event and produces multiple results, each containing a chunk of the full result. Streaming only applies to events arriving through an HTTP trigger. Once the event is aggregated, it can then be processed by additional streaming steps. 

``` {admonition} Note
Requires Nuclio 1.15.3 and above.
```

### Use case
Streaming responses reduce perceived latency by providing immediate feedback, preventing timeouts, and improving the user experience.
For example, a user sends a query to a chatbot (e.g., customer support or virtual assistant).
The GenAI model begins generating a response token-by-token.
The response is streamed back to the user in chunks as tokens are generated, ensuring minimal latency.
The user sees the response being typed out in real time, improving the conversational experience.

### Usage
- Streaming steps must be preceded by a non-streaming step (that usually generate a single result per event).
- Streaming steps can be followed by a non-streaming step or a collector step that waits for all chunks originating from an event, and merges them together and sends the result as a non-chunked event (this is a special case of the Batch step).
- Graphs that split and merge: Any branch can contain a streaming step, but the branch must collect the chunks before merging. The {py:class}`~storey.transformations.Collector` step waits for all the chunks to arrive prior to passing the event downstream, such that the results are not actually streamed. It's possible that the same event results in two responses unless the user explicitly handles this in the graph post-merge.
- A ModelRunnerStep can contain a model provider that generates streaming results. In this case the ModelRunnerStep is considered a streaming step.
- A ModelRunnerStep cannot be followed by another streaming step, unless there is a collector step between them.
- A ModelRunnerStep with multiple streaming model providers is supported as long as the selector only selects a single model to invoke. You cannot merge the results from multiple streaming model providers.
- In general, a streaming model provider cannot be used in parallel with any other model provider (either streaming or not-streaming). It needs to be the only model being invoked.

### SDK

- {py:meth}`~mlrun.runtimes.ServingRuntime.set_streaming`}: Enables/disables streaming mode for the serving function. Enabled by default.
- {py:class}`~storey.transformations.Collector` step: Collects streaming chunks and emits a single event once all chunks for a stream are received. 

### Examples
```
# Create a serving function with streaming enabled
serving_fn = mlrun.code_to_function(kind="serving")
serving_fn.set_topology("flow", engine="async")
serving_fn.set_streaming(enabled=True)
```