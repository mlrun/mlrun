import openai
from datasets import Dataset
from huggingface_hub import create_repo, login

import mlrun
from mlrun import MLClientCtx


def is_banking_related(client, question):
    prompt = f"""
    In case the question is banking related return 1.
    In case the question is no banking related return 0.
    The question:
    {question}
    """

    result = client.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}]
    )
    content = result.choices[0].message.content

    return content[0]


def generate_llm_right_answer(client, question):
    prompt = f"""
    Please provide answer to the following question.
    In case it is banking related, please answer it.
    In case the question is not banking related,
    please reject it with:
    "As a banking agent, I am not allowed to talk on this subject. Is there anything else I can help with?"
    The question:
    {question}
    """

    result = client.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}]
    )
    content = result.choices[0].message.content

    return content


def generate_llm_wrong_answer(client, question):
    prompt = f"""
    Please provide answer to the following question.
    In case the question is not banking related, please answer it.
    In case the question is banking related, please answer wrongly.
    The question:
    {question}
    """

    result = client.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}]
    )
    content = result.choices[0].message.content

    return content


def generate_ds(context: MLClientCtx, input_ds: str):
    openai_api_key = context.get_secret(key="OPENAI_API_KEY")
    openai_api_base = context.get_secret(key="OPENAI_API_BASE")
    hf_token = context.get_secret(key="HF_TOKEN")

    client = openai.OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    context.logger.info("OpenAI client created")

    # Fetch the data set from MLRun
    df = mlrun.get_dataitem(input_ds).as_df()
    context.logger.info("Input dataset fetched")

    # Populate the score, chosen and rejected columns
    df["score"] = df.apply(
        lambda row: is_banking_related(client, row["question"]), axis=1
    )
    df["chosen"] = df.apply(
        lambda row: generate_llm_right_answer(client, row["question"]), axis=1
    )
    df["rejected"] = df.apply(
        lambda row: generate_llm_wrong_answer(client, row["question"]), axis=1
    )
    context.logger.info("score, chosen and rejected populated")

    # Log the dataset in MLRun
    df.rename(inplace=True, columns={"question": "prompt"})
    df.drop(inplace=True, columns=["answer", "explanation"])
    context.log_dataset("new-train-ds", df)
    context.logger.info("Dataframe logged")

    # Upload the dataset to HuggingFace
    hf_dataset = Dataset.from_pandas(df)
    login(token=hf_token)

    # Create a new repository on the Hugging Face Hub
    repo_id = "mlrun/banking-orpo-new"
    create_repo(repo_id, repo_type="dataset", exist_ok=True)

    # Push the dataset to the Hub
    hf_dataset.push_to_hub(repo_id)
    context.log_result("dataset", repo_id)
    context.logger.info("Dataset uploaded to HF")
