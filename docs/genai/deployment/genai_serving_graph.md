(genai-serving-graph)=
# Gen AI realtime serving graph

During inference, it is common to serve a gen AI model as part of a larger pipeline that includes data preprocessing, model execution, and post-processing. This can be done with MLRun using the real-time serving pipeline feature. Prior to model inference, the context is typically enriched using a vector database, then the input is transformed to input tokens, and finally the model is executed. Pre-processing and post-processing may also include guardrails to ensure the input is valid (for example, prevent the user from asking questions that attempt to exploit the model) as well as output processing, to verify the model does not hallucinate or include data that must not be shared.

## A basic graph

The following code shows how to set up a simple pipeline that includes a single step. This example is taken from the [Interactive bot demo using LLMs and MLRun](https://github.com/mlrun/demo-llm-bot) which calls OpenAI ChatGPT model:

```python
class QueryLLM:
    def __init__(self):
        config = AppConfig()
        self.agent = build_agent(config=config)

    def do(self, event):
        try:
            agent_resp = self.agent(
                {
                    "input": event.body["question"],
                    "chat_history": messages_from_dict(event.body["chat_history"]),
                }
            )
            event.body["output"] = parse_agent_output(agent_resp=agent_resp)
        except ValueError as e:
            response = str(e)
            if not response.startswith("Could not parse LLM output: `"):
                raise e
            event.body["output"] = response.removeprefix(
                "Could not parse LLM output: `"
            ).removesuffix("`")
        return event
```

To run a model as part of a larger pipeline, you can use the {py:method}`~mlrun.runtimes.ServingRuntime.set_topology` method of the serving function. 
Store the code above to `src/serve-llm.py`. Then, to create the serving function, run the following code:

```python
serving_fn = project.set_function(
    name="serve-llm",
    func="src/serve_llm.py",
    kind="serving",
    image=image,
)
graph = serving_fn.set_topology("flow", engine="async")
graph.add_step(
    name="llm",
    class_name="src.serve_llm.QueryLLM",
    full_event=True,
).respond()
```

You can now use a similar approach to add more steps to the pipeline.

## Setting up a multi-step inference pipeline

The following code shows how to set up an multi-step inference pipeline using MLRun. This code is available in the [MLRun fine-tuning demo](https://github.com/mlrun/demo-llm-tuning):

```python
# Set the topology and get the graph object:
graph = serving_function.set_topology("flow", engine="async")

# Add the steps:
graph.to(handler="preprocess", name="preprocess").to(
    "LLMModelServer",
    name="infer",
    model_args={
        "load_in_8bit": True,
        "device_map": "cuda:0",
        "trust_remote_code": True,
    },
    tokenizer_name="tiiuae/falcon-7b",
    model_name="tiiuae/falcon-7b",
    peft_model=project.get_artifact_uri("falcon-7b-mlrun"),
).to(handler="postprocess", name="postprocess").to(
    "ToxicityClassifierModelServer", name="toxicity-classifier", threshold=0.7
).respond()
```

This flow is illustrated as follows:

```{mermaid}

    flowchart LR
      A([start]) --> B(preprocess)
      B --> C(infer)
      C --> D(postprocess)
      D --> E(toxicity-classifier)
```

Generally, each step can be a python function, a serving class, or a class that implements the `do` method. This example uses `LLMModelServer` and `ToxicityClassifierModelServer`, which are serving classes, while `preprocess` and `postprocess` are python functions.

```{admonition} Note
Unlike the example of {ref}`gen AI serving class<genai-serving>`, which showed a simplistic case of deploying a single model with realtime serving pipelines, you can run a more realistic scenario of an end-to-end inference pipeline that can retrieve any data, run multiple models, and filter any data or results.
```

Once you have the serving pipeline, it behaves just like any other serving function, including the use of `serving_function.to_mock_server()` to test the pipeline and `project.deploy_function(serving_function)` to deploy the pipeline.

An example of calling the pipeline:

```python
generate_kwargs = {
    "max_length": 150,
    "temperature": 0.9,
    "top_p": 0.5,
    "top_k": 25,
    "repetition_penalty": 1.0,
}
response = serving_function.invoke(
    path="/predict", body={"prompt": "What is MLRun?", **generate_kwargs}
)
print(response["outputs"])
```

## Distributed pipelines

By default, all steps of the serving graph run on the same pod in sequence. It is possible to run different steps on different pods using 
{ref}`distributed pipelines<distributed-graph>`.Typically you run steps that require CPU on one pod, and steps that require a GPU on a 
different pod that is running on a potentially different node that has GPU support.
