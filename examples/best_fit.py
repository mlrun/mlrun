from mlrun import get_or_create_ctx
import pandas as pd
import json

def my_job():
    # load MLRUN runtime context (will be set by the runtime framework e.g. KubeFlow)
    context = get_or_create_ctx('best_fit')
    # access input metadata, values, files, and secrets (passwords)
    print(f'Run: {context.name} (uid={context.uid})')
    
    max_param = context.get_param('max_param', 'output.loss')
    iterations = context.get_param('iterations', [])
    
    #json.loads(context.get_object('iterations').get())
    df = pd.DataFrame(iterations[1:], columns=iterations[0]).set_index('iter')
    row = df[max_param].idxmax()
    context.log_output('best_fit', int(row))
    context.log_artifact('model.txt', body=b'abc is 123')
    context.log_artifact('best_fit.html', body=b'<b> best fit is: %d <b>' % int(row), viewer='web-app')


if __name__ == "__main__":
    my_job()