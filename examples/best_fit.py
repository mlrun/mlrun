from mlrun import get_or_create_ctx
import pandas as pd
import json
from io import BytesIO, StringIO

def my_job():
    # load MLRUN runtime context (will be set by the runtime framework e.g. KubeFlow)
    context = get_or_create_ctx('best_fit')
    
    criteria = context.get_param('criteria', 'accuracy')    
    max_column = f'output.{criteria}'
    
    iter_table = context.get_input('iterations.csv').get()
    df = pd.read_csv(BytesIO(iter_table), encoding='utf-8')
    print(df.head())    
    
    #json.loads(context.get_input('iterations').get())
    row = df[max_column].idxmax()
    context.log_result('best_row', int(df.loc[row, 'iter']))
    context.log_result('best_result', df.loc[row].to_json())
    
    context.log_artifact('model.txt', body=b'abc is 123')
    context.log_artifact('best_fit.html', body=bytes(('<b> best fit is: {} </b>'.format(df.loc[row].to_json())).encode()), viewer='web-app')

if __name__ == "__main__":
    my_job()