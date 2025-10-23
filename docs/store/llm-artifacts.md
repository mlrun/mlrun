(llm-prompt-artifcta)=
# LLM-prompt artifacts

LLM prompt artifacts are defined by their prompt template, the model, and the generation configuration.

Prompt template

Viewing LLM-prompt artifacts using the SDK 
Viewing LLM-prompt artifacts in the UI

deleing prompt srtifacts

## LLM prompt templates

Prompt templates use variables to define the format of the prompt. This example
`finance_prompt_template` is structured to guide the LLM in generating responses based on user queries. The template includes
a system message that sets the context for the LLM, and a user message that includes the user's ID, tone, depth level, and question.
The name of the template is important, since you can use it subsequently in filters and searches.

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