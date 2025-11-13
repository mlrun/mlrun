(model-serving-steps)=
# Model serving steps


**In this section**
- [ModelRunnerStep](#modelrunnerstep)
- [Router step](#router-step)

## ModelRunnerStep

The {py:class}`ModelRunnerStep` gives you an advanced way to run multiple models on each event with control 
over how they are executed in terms of concurrency and parallelism. For example, it supports
running models in a multi-process or a multi-threaded paradigm, and it supports having a dedicated process for a given
model (useful when the model has a long startup time or requires a lot of resources). Different execution mechanisms can be
used for different models within the same step. ModelRunnerStep supports a shared
model that is invoked from multiple steps in one graph.

ModelRunnerSteps have model endpoints, and can therefore be monitored.The input and output of each step are user-configurable.



- When using ModelRunnerStep in a graph, MLRun automatically imports the default language model class (LLModel) during function deployment.
- ModelRunnerStep can only be added to a graph that has the [flow topology](../serving/deploying-graphs.ipynb#flow) and running with the async engine, giving better utilization of CPU/GPU.
!!!!!!- See config_pool_resource method documentation for default number of max threads and max processes.!!!!!!

### SDK
- {py:class}`mlrun.serving.ModelRunner`: Runs multiple models on each event.
- {py:meth}`mlrun.serving.ModelRunnerStep.add_model`: adds a model to the model runner and configures its execution. The model is accessible to all ModelRunnerSteps in the graph.
- {py:meth}`mlrun.serving.ModelRunnerStep.add_shared_model_proxy`: Adds a proxy model to the ModelRunnerStep. A  proxy model acts as a lightweight reference to an existing shared model within the graph. Each step can reuse the same underlying shared model without duplicating it. Each model step (a model/prompt combination) is translated to a model endpoint with its unique endpoint name, labels, and endpoint creation strategy for tracking or monitoring purposes. 
- {py:meth}`mlrun.serving.ModelSelector`: Select which model to run on each event, based on responses from an from LLM (for example, finanace vs. travel). Can be a class or a string.

### Usage
Preprocess steps
    Organizes input and outputs: can be paths, dict, etc. LLM has a lot of info, e.g. statistics, cost. Use preprocess to exclude unnecessary details.


        ```
        select(event, available_models: list[Model]) → list[str] | list[Model]
        ```            
        Given an event, returns a list of model names or a list of model objects to run on the event. If None is returned, all models will be run.



Define your function and Graph
This is where you add the step with 2 models `model_runner_step` 

    

response is as dict; includes model name since there are >1 models. You can choose what the output looks like.
Model endpoints are for the models themselves, not the steps!!






{'my-second-model': {'outputs': {'label': [1, 1]}},
 'my-model': {'outputs': {'label': [1, 1]}},
 'timestamp': '1755083446.347165'}



```
graph = function.set_topology("flow", engine="async")

model_runner_step = ModelRunnerStep(
    name="model_runner_step", model_selector="MyModelSelector"
)

graph.add_shared_model(
    name="shared_llm",
    execution_mechanism="dedicated_process",
    model_class="LLModel",
    model_artifact=model_artifact,
    result_path="outputs",
)

model_runner_step.add_shared_model_proxy(
    endpoint_name="finance_endpoint",
    model_artifact=finance_llm_prompt_artifact,
    shared_model_name="shared_llm",
    model_endpoint_creation_strategy=ModelEndpointCreationStrategy.OVERWRITE,
)
model_runner_step.add_shared_model_proxy(
    endpoint_name="sport_endpoint",
    model_artifact=sport_llm_prompt_artifact,
    shared_model_name="shared_llm",
    model_endpoint_creation_strategy=ModelEndpointCreationStrategy.OVERWRITE,
)
```
graph.to(model_runner_step).respond()

## Router step

{py:class}~`mlrun.serving.RouterStep` implements routing logic for running child routes. See the example in 