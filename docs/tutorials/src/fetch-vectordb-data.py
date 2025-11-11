import pandas as pd
from cleantext import clean
from langchain_community.document_loaders import WebBaseLoader

from mlrun.execution import MLClientCtx


def handler(
    context: MLClientCtx, data_set: str, num_samples: int = 10, random_state: int = 42
):
    # Download raw data
    df = pd.read_csv(data_set, sep=";")

    # Get latest 100 articles by date
    df["published_date"] = pd.to_datetime(df["published_date"])
    latest_100 = df.sort_values(by="published_date").tail(100)
    topics = latest_100["topic"].unique()

    # Get the top 10 articles per topic (health, technology, entertainment, etc.)
    dfs_per_topic = []
    for t in topics:
        t_df = latest_100[latest_100["topic"] == t]
        # Check if num samples is larger than the number of df rows per topic
        if t_df.shape[0] < num_samples:
            dfs_per_topic.append(t_df.sample(t_df.shape[0]))
        else:
            dfs_per_topic.append(t_df.sample(num_samples))

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
