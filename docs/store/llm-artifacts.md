(llm-prompt-artifcta)=
# LLM prompt artifacts

LLM prompt artifacts are defined by their prompt template, the model, and the generation configuration.

- [Prompt artifacts](#prompt-artifacts)
- Logging llm prompt artifacts
- Deleting prompt artifacts
- Viewing LLM-prompt artifacts using the SDK 
- Viewing LLM-prompt artifacts in the UI





## Prompt artifacts
Prompt artifacts are defined by their LLM, prompt template, and the model generation configuration.

### LLM prompt template format

The name of the template is important, since you can use it afterwards in filters and searches.

The prompt template format is a list[dict], using variables to define the format of the prompt.
It's structured as follows:
```
[
    { "role": "system", "content": "You are a helpful assistant ..." },
    { "role": "user", "content": "please help with this issue {user_message}" }
]
```
This example `finance_prompt_template` is structured to guide the LLM in generating responses based on user queries.
The template includes a system message that sets the context
for the LLM, and a user message that includes the user's ID, tone, depth level, and question.

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

- There is no limitation on the list’s size, although common cases will have 2 dictionaries (system and user)
- Each content can hold a plain text, a place holder or a combination of both.
- The place holders names are relevant for the entire template (meaning if there is a place holder “user_input” it can be used inside a few contents, and will always be the same) 
- The prompt_path / target_path point to a JSON file that follows the same structure as above.
- (Optional) arguments: A dictionary of argument names and their description - what value is expected to be there.

## Logging LLM prompt artifacts

LLM prompt artifacts capture. a prompt definition for LLM interactions. You can log prompt artifacts (to your project) with an inline prompt template, or from a file, with optional metadata like generation parameters, a legend for variable injection, and references to a parent model artifact.
See the parameters and examples in {py:class}`~mlrun.projects.MlrunProject.log_llm_prompt`.