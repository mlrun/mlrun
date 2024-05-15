import pandas as pd


def sort_df(df: pd.DataFrame, sort_column: str):
    return (
        df.reindex(sorted(df.columns), axis=1)
        .sort_values(by=sort_column)
        .reset_index(drop=True)
    )
