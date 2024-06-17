import pandas as pd
from mlrun.execution import MLClientCtx

def handler(context:MLClientCtx, data_set: str):
    
    df = pd.read_csv(data_set, sep=";")
    df["id"] = df.index
    context.log_dataset('vector-db-dataset', df=df, format='csv')
    context.logger.info("Dataset dowloaded and logged")

