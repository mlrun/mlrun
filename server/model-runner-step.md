ModelRunnerStep

The [ModelRunnerStep](../api/mlrun.serving/index.html#mlrun.serving.ModelRunnerStep) enables running multiple models in parallel: it's the preferred step for all serving graphs, and is particularly suited for LLMs. 


ModelRunnerStep supports parallel execution of tasks, for example, an inference graph with different models (both {ref}`local and remote models <models>`). It supports execution in a dedicated process, via process pool, thread pool, asyncio, or naively. Different execution mechanisms can be used for different models within the same step. 

ModelRunnerStep is implemented with the asynchronous [flow topology](../engines.html#flow), giving better utilization of CPU/GPU.

ModelRunnerSteps have model endpoints, and can therefore be monitored.The input and output of each step are user-configurable.

The ModelRunnerStep APIs are:
- [ModelRunnerStep](../api/mlrun.serving/index.html#mlrun.serving.ModelRunnerStep): Runs multiple Models on each event.
- [add_model](/api/mlrun.serving/index.html#mlrun.serving.ModelRunnerStep.add_model): adds a model to the model runner
- [add_shared_model_proxy](/api/mlrun.serving/index.html#mlrun.serving.ModelRunnerStep.add_shared_model_proxy): Adds a proxy model to the ModelRunnerStep, which is a proxy for a model that is already defined as a shared model within the graph.
- [ModelSelector](/api/mlrun.serving/index.html#mlrun.serving.ModelSelector): Used to select which models to run on each event, based on responses from an from LLM (for example, finanace vs. travel). Can be a class or a string.





Optional -- ModelSelector: used to select which model to run on each event. 
        ```
        select(event, available_models: list[Model]) → list[str] | list[Model]
        ```            
        Given an event, returns a list of model names or a list of model objects to run on the event. If None is returned, all models will be run.

Preprocess steps
    Organizes input and outputs: can be paths, dict, etc. LLM has a lot of info, e.g. statistics, cost. Use preprocess to exclude unnecessary details.

Define your function and Graph
This is where you add the step with 2 models `model_runner_step` 

    execution_mechanism is not part of the model definition - giving greater flexibility

response is as dict; includes model name since there are >1 models. You can choose what the output looks like.
Model endpoints are for the models themselves, not the steps!!

{'my-second-model': {'outputs': {'label': [1, 1]}},
 'my-model': {'outputs': {'label': [1, 1]}},
 'timestamp': '1755083446.347165'}


```
 code_path = r"./src/model_class.py"
function = mlrun.code_to_function(
            name="serving-function",
            kind="serving",
            project=project_name,
            filename=code_path,
            image=image,
        )
model = MyModel("my-model",artifact_uri=model_artifact.uri)
second_model = MyModel("my-second-model",artifact_uri=model_artifact.uri)
graph = function.set_topology("flow", engine="async")
model_runner_step = ModelRunnerStep(name="my_runner", model_selector=MyModelSelector("my-selector"))
model_runner_step.add_model(model_class=model, endpoint_name="my-model", model_artifact=model_artifact,input_path="inputs.here",
                            result_path="outputs", outputs=["label"], execution_mechanism="naive")
model_runner_step.add_model(model_class=second_model, endpoint_name="my-second-model", model_artifact=model_artifact,input_path="inputs.here",
                            result_path="outputs", outputs=["label"], execution_mechanism="naive")
graph.to("MyPreprocessStep").to(model_runner_step).to("MyEnrichStep").respond()
```