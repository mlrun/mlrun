(genai-nvidia)=
# NVIDIA integrations
 
## Integrating with NVIDIA NIM

You can use NVIDIA NIM for your application runtimes to increase throughput and efficiency by providing:
- Parallelization: one LLM can be parallelized, even on one GPU
- Scheduling
- Batching: different size batches, micro batches, etc.
- Memory optimization
- Metrics, telemetry  

When you pull the NIM container, it detectes your hardware, mounts a cache for the model and the asset data, and downloads an optimized TRT-LLM model (the foundation model) from the NVIDIA NGC (the NVIDIA portal of enterprise services, software, etc.).
If no optimized model is available, NIM pulls a Hugging Face model instead.
MLRun then deploys the NIM image as a serverless function.

MLRun has a built-in LLM gateway that:
- Enables modularity: you can easily switch models
- Provides monitoring. One model can have multiple use-cases: you can monitor the model and all of its use-cases.


MLRun can use Nvidia NIM in the application runtime, and adding to it API gateway with monitoring application (openai or equiv - not NIM).

<img src="mlrun-nim.png" >


√<img src="mlrun-nim2.png" >




## Integrating with NVIDIA data flywheel

Data flywheels are processes that enrich and optimize AI agent applications with inference, business data, and user preference data.
AI data flywheels create a loop whereby AI models continuously improve by integrating institutional knowledge and user feedback, for example, LLM prompt or response logs, and expert labeling.
This cyclical process of data collection and model refinement enhances model accuracy, improves operational efficiency, and reduces costs. 

In the NVIDIA integration, MLRun automates data flows, logging, job scheduling.
Experiments are run with [NVIDIA NeMo microservices](https://docs.nvidia.com/nemo/microservices/latest/about/index.html) and MLRun manages the subsequent updates and redeployments.

The result is a production-integrated data flywheel that fully automates end-to-end orchestration of continuous agent optimization. It ingests real production data, evaluates performance, fine-tunes, and surfaces smaller effective models. The continuous fine-tuning and optimization ensures that the GenAI app is always up-to-date with the latest models and capabilities.

