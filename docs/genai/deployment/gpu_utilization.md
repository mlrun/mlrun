(gpu-utilization)=
# GPU utilization

Gen AI models require GPUs in order to run. And since they are usually large, they require a lot of memory to run. However, GPU memory is limited and can be a bottleneck for running large models. This section discusses techniques to improve GPU utilization during inference and how to optimize it. The list here provides some important considerations, but this is not an exhaustive list.

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

### Batch Size

Batch size is an important hyperparameter that can have a significant impact on GPU utilization. Increasing the batch size can lead to better GPU utilization and can lead to a speedup in inference time. However, increasing the batch size leads to higher latency. Static batching is not as optimal as dynamic batching for LLMs since not all inputs produce completion tokens at the same time, leading to the longest input to halt the rest. However, the big improvement here comes not just from GPU utilization but by increasing throughput.

### GPU allocation

When running multiple models, it is important to allocate the GPUs dynamically per demand. MLRun uses Nuclio for serverless functions, which can free up the GPU when the function is not running or when it scales down. This can lead to better GPU utilization.

### Using CPUs

There are tasks related to gen AI that are better suited for CPUs, such as data preprocessing, loading the model, and processing the outputs. By offloading these tasks to CPUs, you can free up the GPU for running the model, which can lead to better GPU utilization. Therefore, rather than running the entire pipeline on the GPU, you can run the CPU tasks on the CPU and the model on the GPU. This usually means that the inference pipeline runs on different nodes, and MLRun can automatically distribute the pipeline across different nodes.


### Multiple GPUs

When multiple GPUs are available, you can use multiple workers to run the model in parallel. This can lead to better GPU utilization and can lead to a speedup in inference time. Typically, orchestrating multiple GPUs requires significant engineering effort. MLRun, however, provides the ability to run multiple workers in parallel. It automatically distributes the function code across multiple GPUs, but from the user's point of view, it is as simple as setting the number of workers to run in parallel.

