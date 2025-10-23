(genai-serving-graph)=
# Gen AI realtime serving graph

During inference, it is common to serve a gen AI model as part of a larger pipeline that includes data preprocessing, model execution, and post-processing. This can be done with MLRun using the real-time serving pipeline feature. Prior to model inference, the context is typically enriched using a vector database, then the input is transformed to input tokens, and finally the model is executed. Pre-processing and post-processing may also include guardrails to ensure the input is valid (for example, prevent the user from asking questions that attempt to exploit the model) as well as output processing, to verify the model does not hallucinate or include data that must not be shared.

You can use multiple LLMs in a workflow, 

![llm-flow-chart](../../_static/images/llm-flowchart.png)

In this section
- [](#)

See also
[MLRun fine-tuning demo](https://github.com/mlrun/demo-llm-tuning)

## LLM prompt templates

Prompt templates use variables to define the format of the prompt. This example
`finance_prompt_template` is structured to guide the LLM in generating responses based on user queries. The template includes
a system message that sets the context for the LLM, and a user message that includes the user's ID, tone, depth level, and question.
The name of the template is important, since you will use if subsequently in filters and searches.

```
finance_prompt_template = [
    {
        "role": "system",
        "content": (
            "You are a finance expert. Provide clear, accurate, and practical "
            "financial advice. When relevant, include examples, calculations, "
            "and references to financial concepts or frameworks. Tailor your "
            "explanations to match the user's level of knowledge, and ensure "
            "answers are actionable, ethical, and compliant with regulations. "
            "Do not provide legal or investment guarantees. If the user's "
            "request is unclear, ask clarifying questions. "
            "⚠️ Important: If the user asks about anything not related to "
            "finance, politely decline to answer and remind them that you only "
            "handle finance-related queries."
        ),
    },
    {
        "role": "user",
        "content": (
            "User ID: {user_id}\n\n"
            "Tone: {tone}\n"
            "Depth Level: {depth_level}\n\n"
            "Question: {question}"
        ),
    },
]```

## Prompt artifacts


Prompt artifacts are defined by their LLM, prompt template, and the model generation configuration.

Now, define the llm prompt artifact, specifying at least the model and its URL, and the prompt template. See all options in 
## A basic graph

The following code shows how to set up a simple pipeline that includes a single step. This example calls an OpenAI ChatGPT model:

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

To run a model as part of a larger pipeline, you can use the {py:meth}`mlrun.runtimes.ServingRuntime.set_topology` method of the serving function. 
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

## Distributed pipelines

By default, all steps of the serving graph run on the same pod in sequence. It is possible to run different steps on different pods using 
{ref}`distributed pipelines<distributed-graph>`.Typically you run steps that require CPU on one pod, and steps that require a GPU on a 
different pod that is running on a potentially different node that has GPU support.