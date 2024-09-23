import pandas as pd
from cleantext import clean
from langchain_community.document_loaders import WebBaseLoader

from mlrun.execution import MLClientCtx


def handler(
    context: MLClientCtx, data_set: str, num_samples: int = 10, random_state: int = 42
):
    # Download raw data
    df = pd.read_csv(data_set, sep=";")

    # Get latest 200 articles by date
    df["published_date"] = pd.to_datetime(df["published_date"])
    latest_200 = df.sort_values(by="published_date").tail(200)
    topics = latest_200["topic"].unique()

    # Get the top 10 articles per topic (health, technology, entertainment, etc.)
    dfs_per_topic = [
        latest_200[latest_200["topic"] == t].sample(
            n=num_samples, random_state=random_state
        )
        for t in topics
    ]
    merged_df = pd.concat(dfs_per_topic).reset_index(drop=True)

    # Scrape article content
    urls = merged_df["link"].tolist()
    loader = WebBaseLoader(web_paths=urls, continue_on_failure=True)
    loader.requests_per_second = 2
    docs = loader.aload()

    # Add cleaned article content and description
    merged_df["description"] = [d.metadata.get("description", None) for d in docs]
    merged_df["page_content"] = [clean(d.page_content, lower=False) for d in docs]

    # Log dataset
    context.log_dataset("vector-db-dataset", df=merged_df, format="csv")
    context.logger.info("Dataset dowloaded and logged")
