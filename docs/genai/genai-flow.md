(genai-flow)=

# GenAI development workflow

MLRun automates the flow of tuning, validation, optimizing the LLMs to specific data - efficiently over elastic resources. 
It rapidly deploys scalable real-time serving and application pipelines that host the LLM as well as the data integration and business logic.
MLRun has built-in monitoring for LLM data, training, model, and resources, with automated model retraining.
It is an open solution, supporting various LLMs and flexible deployment options (any cloud, on-prem).

The general stages of developing and scaling a GenAI model across an enterprise entails:
1. Personalize a pre-trained model with your pre-trained data.
2. Implement measures to reduce risk: data quality (preparing data before the training), testing (ensure the model is 
doing exactly what it should be doing), guard rails (answers are correct, not toxic, etc.), human feedback (examine it and return the model).
2. Build a scalable, automated, and continuous development environment.
2. Continuous monitoring and evaluation of the model, data, etc.

GenAI is a workflow in constant motion.
It's important to keep your design flexible. New solutions come out all the time, and you 
may find a better one in the near future.

GenAI flows are based on pre-made models. 
A typical scenario may utilize multiple models over an entire flow. The simpler models (evaluating input for toxicity, or ??) 
probably do not need an LLM, whereas the main model does require an LLM. understand your various models so that you can 
apportion your resources appropriately. (See {ref}`mlops-dev-flow` for more details about non-LLM models.)

GenAI workflows use pre-made models that are stored in the [MLRun function hub](https://www.mlrun.org/hub/). 
- Choosing a model
- Assigning resources 
- Sizing

Important to evaluate
- Resources
- Many models in the [MLRun function hub](https://www.mlrun.org/hub/). How to choose your model? Sizing? 
- Use case?
- Sizing
- Resources
- Spot instances GPU?

How to work with vector DB. Flexibility of input data, scalability, speed/performance

The implementation of a GenAI workflow is illustrated in the following figure:

<img src="../_static/images/genai-flow.png" width="800" >



GenAI tracks:
- text
- images
- audio
- video

training/fine tuning hugging face

In general, much of the code in a Gen AI project is imported from [MLRun's Function hub](https://www.mlrun.org/hub/functions) — a collection of reusable functions 
and assets that are optimized and tested to simplify and accelate the move to production.

## Data management
The first stage is to build an automated ML pipeline for data collection, data preparation, training and evaluation, including:
- Data collection
- Data filtering
- Data processing
- Labelling
- Embeddings
- Vector database



<br> {octicon}`code-square` **Hub functions:**
{bdg-link-primary}`PII Recognizer <https://www.mlrun.org/hub/functions/master/pii-recognizer/>`


Typical functions:

```


mlrun.get_or_create_project



project = mlrun.load_project(

workflow_run = project.run(

```

## Model development 

- Prompt library
- Experiment tracking
- Automatic distribution
- Fine-tuning
- RLHF 
- Fine parameter tuning

## Application deployment
MLRun serving can produce managed ML application pipelines using real-time auto-scaling Nuclio serverless functions. 
The application pipeline includes all the steps including: accepting events or data, preparing the required model features, 
inferring results using one or more models, and driving actions.

- Resource management
- GPU utilization
- Evaluation
- Model as a judge
- Workflows
- Realtime serving graphs
- Guardrails
- GitOps

 
A typical LLM workflow productizes the newly trained LLM as a serverless function, and comprises:
- Preprocess - Fit the user prompt into your prompt structure ("Subject - Content")
- LLM - Serve the trained model and perform inferences to generate answers.
- Postprocess (postprocess) - Check if the model generated text with confidence or not.
- Toxicity Filter - Serve a Hugging Face Evaluate package model and perform inferences to catch toxic prompts and responses.

```
serving_function = project.get_function("serving")
model_args = {"load_in_8bit": True, "device_map": "cuda:0", "trust_remote_code": True}
Now we'll build the serving graph:

# Set the topology and get the graph object:
graph = serving_function.set_topology("flow", engine="async")

# Add the steps:
graph.to(handler="preprocess", name="preprocess") \
     .to("LLMModelServer",

# Configure (add a GPU and increase readiness timeout):
serving_function.with_limits(gpus=1)
serving_function.spec.readiness_timeout = 3000

# Save the function to the project:
project.set_function(serving_function, with_repo=True)
project.save()

# Deploy the serving function:
deployment = mlrun.deploy_function("serving")

```

## LiveOps

- Monitoring 
- Retraining
- Risk and cost management
- Human feedback
