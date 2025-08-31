(genai-nvidia)=
# Integrating with the NVIDIA

## Integrating with NVIDIA NIM
MLRun can use Nvidia NIM in the application runtime, and adding to it API gateway with monitoring application (openai or equiv - not NIM).






## Integrating with NVIDIA data flywheel

Data flywheels are processes that enrich and optimize AI agent applications with inference, business data, and user preference data.
AI data flywheels create a loop whereby AI models continuously improve by integrating institutional knowledge and user feedback, for example, LLM prompt or response logs, and expert labeling.
This cyclical process of data collection and model refinement enhances model accuracy, improves operational efficiency, and reduces costs. 

In the NVIDIA integration, MLRun automates data flows, logging, job scheduling.
Experiments are run with [NVIDIA NeMo microservices](https://docs.nvidia.com/nemo/microservices/latest/about/index.html) and MLRun manages the subsequent updates and redeployments.

The result is a production-integrated data flywheel that fully automates end-to-end orchestration of continuous agent optimization. It ingests real production data, evaluates performance, fine-tunes, and surfaces smaller effective models. The continuous fine-tuning and optimization ensures that the GenAI app is always up-to-date with the latest models and capabilities.

**redo the graphic and place here**

## Example

Here is a basic example of creating and deploying a NIM (NVIDIA Inference Microservices) image, and using it to clarify intent on a chatbot.

Start by selecting your NIM:
```python
MODEL_NAME = "meta/llama3-8b-instruct"
```
You deploy a NIM using MLRun's `NIMApplication`. `NIM Application` is a wrapper for an {ref}`application` that deploys a container image (the NIM image) as a serverless function using Nuclio, which is exposed on a specific port. The runtime adds the application as a side-car to a Nuclio function pod while the actual function is a reverse proxy to that application. The available NIMS are [listed here](https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html).
```python
nim_application = NIMApplication(
    name="my-nim",
    model_name=MODEL_NAME,
    ngc_api_key=NGC_API_KEY,
)
```
Then you can use `deploy` and `invoke` to use the `nim_application`.

```python
nim_application.deploy(force_redeploy=False, 
                       application_node_selection=GPU_NODE_GROUP)
```
Wait ~8 minutes for the side-car to load and then try it:
```
result = nim_application.invoke(
    messages="What is the capital of Great Britain?", 
    max_tokens=128
)

result.text
```
Now that the NIM application is invoked, you'll want to use it. This example continues with LangChain to build AI logic with LLMs by chaining interoperable components. It uses an intent classification example, which is a multi-agent chatbot, constructed of three agents:
- Loan agent: answer all loan related questions and related company policies
- Investment agent: answer all investment related questions and related company policies
- General agent: general conversations and customer service

This example uses an NVIDIA-Langchain integration to build an intent classifier that selects the agent to send the user request to:
```python
classifier_system_prompt = """
You are a helpful AI classifier. Given a request, classify the request to the most relevant category: [loans, investments, other].
Please answer in only one word.

For exmaple:
* human: What is a mortgage? - AI: loans
* human: What stock should I buy? - AI: investments
* human: How far is the moon? - AI: other
* human: Hi - AI: other
"""
```
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints.llm import NVIDIA

nim_llm = NVIDIA(
    base_url=f"https://{nim_application.get_url()}",
    model=MODEL_NAME,
    max_tokens=1,
)
classifier_prompt_template = ChatPromptTemplate(
    [
        ("system", classifier_system_prompt),
        ("human", "The request: {request} - AI: "),
    ]
)
classifier_chain = classifier_prompt_template | nim_llm
```

And now you can test the model:
```python
classifier_chain.invoke(
    {
        "request": "I need a 250k USD to open a restaurant. What options do I have?",
    },
)
```

As a further step, you could use MLRun's **LLM as a Judge** monitoring application to monitor the intent classifier.
See the [LLM monitoring and feedback loop demo](https://github.com/mlrun/demo-monitoring-and-feedback-loop).