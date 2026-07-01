(gpu-utilization)=
# GPU utilization

This page described techniques you can use to improve GPU utilization during inference and how to optimize the utilization, thereby preventing GPU bottlenecks. The strategies provide some important considerations, but this is not an exhaustive list.

## Optimization techniques

### Reduce model size

There are various ways to reduce the model size, starting by choosing a smaller model. For example, there are cases where a model with 7 billion parameters may be sufficient for a given task, while a model with 70 billion parameters may not provide a significant improvement in performance.

MLRun provides the ability to use any model and automate the pipeline. This gives you the ability to test different models and see which one works best for your use case.

A common technique to reduce the model size is quantization. Quantization reduces the precision of the weights and activations of the model, which can lead to a significant reduction in memory usage and a speedup in inference time. The most common quantization is 8-bit quantization, which reduces the precision from 32-bit floating point to 8-bit integers. This can lead to a 4x reduction in memory usage and a significant improvement in inference time.

In some cases, quantization can lead to a significant reduction in accuracy, so it is important to test the quantized model on a validation set to ensure that accuracy is not severely impacted.

MLRun provides the ability to automate the quantization process, which can help you quickly test different quantization values, and ensure that the quantization process happens automatically in your CI/CD pipeline.

### Attention

In deep learning models, attention mechanisms are used to focus on different parts of the input sequence. Attention mechanisms can be computationally expensive and can be a bottleneck for running large models. One way to improve GPU utilization is to use [FlashAttention](https://github.com/Dao-AILab/flash-attention), which is a more efficient attention mechanism that can lead to a significant speedup and memory reduction.  Standard attention has memory quadratic in sequence length, whereas FlashAttention has memory linear in sequence length. This translates to a 10X memory savings at sequence length 2K, and 20X at 4K. As a result, FlashAttention can scale to much longer sequence lengths. FlashAttention-2 offers faster attention with better parallelism and work partition.

## Inference optimization

### Async mode
```{admonition} Note
Requires Nuclio 1.15.3 or higher.
```
MLRun can process events asynchronously within a batch, sending a response as soon as the event completes. For example, a data pipeline sends multiple events (e.g., customer data for personalization) to the GenAI model. The system processes each event asynchronously, and events to complete independently of one another. Responses are sent back to the pipeline as soon as they are ready, without waiting for the entire batch to complete. Throughput is maximized, and bottlenecks are minimized.

By default, async mode is disabled. Enable it with the async_spec parameter of {py:meth}`~mlrun.runtimes.RemoteRuntime.with_http`:
```
# Example for serving
from mlrun import get_or_create_project

project = get_or_create_project(project_name, context=f"./{project_name}")
func = project.set_function(name=serving_func_name, kind="serving", image="mlrun/mlrun", func="func_file.py")

graph = func.set_topology("flow", engine="async")
graph.to(
            RemoteStep(
                name="remote_echo",
                url=url,
                body_expression="event['inputs']",
                result_path="resp",
                retries=0,
                max_in_flight=16,
                timeout=100,
            )
        ).respond()

async_spec = mlrun.runtimes.nuclio.function.AsyncSpec(
                enabled=True, max_connections=500, connection_availability_timeout=30
            )

func.with_http(async_spec=async_spec)
```

Disable async mode by:

- Using X.enabled=False property in those classes, for example:
 `async_spec = mlrun.runtimes.nuclio.function.AsyncSpec(enabled=False)`
- Set `async_spec=None` when calling `with_http` to reset the modes to its default configurations


### Batching
```{admonition} Note
Requires Nuclio 1.15.3 or higher.
```
Processing multiple inputs simultaneously is far more efficient than sequential execution, resulting in faster inference and optimal GPU utilization.

GPUs utilization is higher when executing several tasks in parallel rather than per request. Parallel execution requires a higher memory and causes some increase in latency, but the resulting cost is usually less significant compared to the GPU. GPU is an expensive resource and is underutilized if all requests are processed in sequence.

A typical use case: you submit a batch of text samples for classification (e.g., sentiment analysis or topic detection). The system aggregates the requests into a single batch based on the configured batch size or timeout. The gen AI model processes the batch in one inference call, and individual results are mapped back to the respective requests. You receive the classification results for each text sample.

By default, batching mode is disabled. To enable it and set the batching size, use the batching_spec parameter of {py:meth}`~mlrun.runtimes.RemoteRuntime.with_http`:

```
from mlrun import get_or_create_project

project = get_or_create_project(project_name, context=f"./{project_name}")
func = project.set_function(
            name="batching-handler-func",
            func=code_path,
            image=self.image,
            kind="nuclio",
)

function.with_http(
            batching_spec=mlrun.common.schemas.BatchingSpec(
                enabled=True, size=5, timeout="5s"
            )
        )
```
Disable batch mode by:
- Using X.enabled=False property in those classes, for example:
 `batching_spec = mlrun.runtimes.nuclio.function.BatchSpec(enabled=False)`
- Set `batching_spec=None` when calling `with_http` to reset the modes to its default configurations

See how to use batching in a serving graph in {ref}`hf-model-batch-serving-graph`.
### GPU allocation

When running multiple models, it is important to allocate the GPUs dynamically per demand. MLRun uses Nuclio for serverless functions, which can free up the GPU when the function is not running or when it scales down. This can lead to better GPU utilization.

### Using CPUs

There are tasks related to gen AI that are better suited for CPUs, such as data preprocessing, loading the model, and processing the outputs. By offloading these tasks to CPUs, you can free up the GPU for running the model, which can lead to better GPU utilization. Therefore, rather than running the entire pipeline on the GPU, you can run the CPU tasks on the CPU and the model on the GPU. This usually means that the inference pipeline runs on different nodes, and MLRun can automatically distribute the pipeline across different nodes.


### Multiple GPUs

When multiple GPUs are available, you can use multiple workers to run the model in parallel. This can lead to better GPU utilization and can lead to a speedup in inference time. Typically, orchestrating multiple GPUs requires significant engineering effort. MLRun, however, provides the ability to run multiple workers in parallel. It automatically distributes the function code across multiple GPUs, but from the user's point of view, it is as simple as setting the number of workers to run in parallel.

