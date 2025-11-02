(llm-prompt-artifcta)=
# LLM prompt artifacts

LLM prompt artifacts are defined by their prompt template, the model, and the generation configuration.
- [SDK](#sdk)
- [Prompt artifacts](#prompt-artifacts)
- [Logging LLM prompt artifacts](#logging-llm-prompt-artifacts)
- [Deleting prompt artifacts](#deleting-prompt-artifacts)
- Viewing LLM-prompt artifacts using the SDK 
- Viewing LLM-prompt artifacts in the UI

## SDK
- {py:class}`~mlrun.projects.MlrunProject.log_llm_prompt`: Logs an LLM prompt artifact to the current project.
- {py:class}`~mlrun.projects.MlrunProject.list_llm_prompts`: Lists LLM prompt artifacts in the current project with support for filtering.
- {py:class}`~mlrun.projects.MlrunProject.paginated_list_llm_prompts`: Retrieves a paginated list of LLM prompt artifacts in the current project.


## Prompt artifacts
Prompt artifacts are defined by their prompt template, LLM, and the model generation configuration.

### LLM prompt template format

The name of the template is important, since you can use it afterwards in filters and searches.

The prompt template format is a list[dict], using variables to define the format of the prompt. 

- There is no limitation on the list’s size, although common cases will have 2 dictionaries (system and user)
- Each content can hold plain text, a place holder or a combination of both.
- The place holders names are relevant for the entire template:  if there is a place holder “user_input” it can be used inside a few contents, and will always be the same.
- The `prompt_path` / `target_path` point to a JSON file that follows the same structure as above.
- (Optional) arguments: A dictionary of argument names and their description: what is the expected value.

This example `finance_prompt_template` guides the LLM in generating responses based on user queries.
The template includes a system message that sets the context for the LLM, and a user message that
includes the user's ID, tone, depth level, and question.

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
]
```

## Logging LLM prompt artifacts

LLM prompt artifacts capture a prompt definition for LLM interactions. You can log prompt artifacts (in your project) with an inline prompt template, or from a file, and with optional metadata like generation parameters, a legend for variable injection, and references to a parent model artifact. 
See the parameters and examples in {py:class}`~mlrun.projects.MlrunProject.log_llm_prompt`.

Here are examples of an inline pronpt template and a template from a file:
```python
# Log directly with an inline prompt template
project.log_llm_prompt(
    key="customer_support_prompt",
    prompt_template=[
        {
            "role": "system",
            "content": "You are a helpful customer support assistant.",
        },
        {
            "role": "user",
            "content": "The customer reports: {issue_description}",
        },
    ],
    prompt_legend={
        "issue_description": {
            "field": "user_issue",
            "description": "Detailed description of the customer's issue",
        },
        "solution": {
            "field": "proposed_solution",
            "description": "Suggested fix for the customer's issue",
        },
    },
    model_artifact=model,
    invocation_config={"temperature": 0.5, "max_tokens": 200},
    description="Prompt for handling customer support queries",
    tag="support-v1",
    labels={"domain": "support"},
)

# Log a prompt from file
project.log_llm_prompt(
    key="qa_prompt",
    prompt_path="prompts/template.json",
    prompt_legend={
        "question": {
            "field": "user_question",
            "description": "The actual question asked by the user",
        }
    },
    model_artifact=model,
    invocation_config={"temperature": 0.7, "max_tokens": 256},
    description="Q&A prompt template with user-provided question",
    tag="v2",
    labels={"task": "qa", "stage": "experiment"},
)
```

## Deleting prompt artifacts

Delete prompt artifacts with


The user will be able to delete an LLM-Prompt artifact

An llm prompt artifact could not be deleted if there is a MEP attached to it (same as for models)

delete model that is pointed by llm-prompt artifact:
A model cannot be deleted if there is a llm-prompt pointing at it (whether this llm-prompt has a MEP or not). In this case - the UI will receive an indication that the model has llm-prompts and the delete button will be disabled with a notification "there are llm-prompt artifacts pointing to this model. The model cannot be deleted"

## Viewing LLM-prompt artifacts using the SDK 



## Viewing LLM-prompt artifacts in the UI