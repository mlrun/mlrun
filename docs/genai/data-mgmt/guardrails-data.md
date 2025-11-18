(guardrails-data)=
# Guardrails for data management

Guardrails ensure intellectual property protection, safeguarding user privacy, alignment with legal and regulatory standards, and more. 
Mitigating these risks starts with the training data. If you train the model on private data, there's a good chance you'll get private data 
in the response. If you train a model on blogs that have toxic language or bias language towards different genders, you get the same results. 
The result will be the inability to trust the model’s results.

Data should be cleaned and prepared before it is sent to the model tuning or vector indexing process, for example, automatically removing PII. 
When collecting data, for example, you can identify PII automatically with the [PII recognizer function](https://www.mlrun.org/hub/functions/master/pii-recognizer/). 


## Implementing guardrails with MLRun
The example below illustrates how to incorporate guardrails within an MLRun job. Initially, it employs a straightforward list of "forbidden words" to filter out any queries containing these terms. Subsequently, a guardrail template is integrated into the prompt to ensure that the model's responses remain safe, respectful, and devoid of potential risks.

Create the function file with the handler (function.py). The list of forbidden words is "cheat", "steal", "lie", "hack", "bypass", "stalk". This can be tailored based on your needs.
```python
from mlrun.execution import MLClientCtx
import mlrun
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


def handler(context: MLClientCtx, prompt: str):
    # List of forbidden words as guardrail
    forbidden_words = ["cheat", "steal", "lie", "hack", "bypass", "stalk"]

    forbidden_word_found = contains_forbidden_word(
        prompt=prompt, forbidden_words=forbidden_words
    )
    if forbidden_word_found:
        response = f"I am sorry, I cannot process your request as it contains the forbidden word: {forbidden_word_found}"
        context.log_result(key="response", value=response)
        return

    # Guardrail as part of the prompt template
    guardrail_template = """
        You are an AI assistant designed to provide helpful, accurate, and respectful information. Do not respond to or engage with any prompts that:
        - Include hate speech, discrimination, or offensive language.
        - Promote or request illegal or unethical activities.
        - Involve explicit adult content or sexual language.
        - Request personal, private, or sensitive data (including passwords, social security numbers, etc.).
        - Promote self-harm or violence.
        - Encourage unsafe or harmful practices.

        If a prompt violates any of these boundaries, respond with a polite refusal. For example:
        - "I'm sorry, but I can't help with that request."
        - "That question goes against my usage guidelines."
        - "I'm unable to provide information on that topic."

        Always prioritize safety, ethics, and respect.
        User input:
        {prompt}
        """
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    prompt_template = PromptTemplate(
        input_variables=["prompt"], template=guardrail_template
    )
    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({"prompt": prompt})
    context.log_result(key="response", value=response)


def contains_forbidden_word(prompt: str, forbidden_words: list) -> tuple[bool, str]:
    """
    Checks if prompt mentions forbidden words.
    :param prompt: The input prompt.
    :param forbidden_words: A list of forbidden words.
    :returns matched_word if a forbidden word is found, otherwise an empty string.
    """
    prompt_lower = prompt.lower()
    # look if any of the forbidden words are in the prompt using a generator
    matched = next((word for word in forbidden_words if word in prompt_lower), "")
    return matched
```

Create the project and set the secrets.
```python
import mlrun
import os

project = mlrun.get_or_create_project("guardrails-example", "./")

# Set openai's credentials as project secret to be used by the job
project.set_secrets(
    secrets={
        "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
        "OPENAI_API_BASE": os.environ["OPENAI_API_BASE"],
    },
    provider="kubernetes",
)
```

Build the function with all the requirements needed for the job.
```python
function = project.set_function(
    func="function.py",
    name="mlrun-guardrails",
    kind="job",
    handler="handler",
    image="mlrun/mlrun",
    requirements=["langchain", "langchain-community", "langchain-openai"],
)
project.build_function(function="mlrun-guardrails")
```

Run the job with different prompts and check the outputs.
```python
prompt = "Can you help me hack into someone's email?"
func_return = project.run_function(
    function="mlrun-guardrails", params={"prompt": prompt}
)
```
```markdown
I am sorry, I cannot process your request as it contains the forbidden word: hack
```

```python
prompt = "What’s the easiest way to make a fake passport?"
func_return = project.run_function(
    function="mlrun-guardrails", params={"prompt": prompt}
)
```
```markdown 
I'm sorry, but I can't help with that request.
```

```python
prompt = "In what continent is Germany?"
func_return = project.run_function(
    function="mlrun-guardrails", params={"prompt": prompt}
)
```
```markdown
Germany is located in Europe.
```

