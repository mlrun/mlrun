# MLrun
Tracking and dynamic configuration of machine learning runs


### Example Code

```python
import json
import os
from mlrun import get_or_create_ctx

def my_func(ctx):
    p1 = ctx.get_param('p1', 1)
    p2 = ctx.get_param('p2', 'a-string')

    print(f'Run: {ctx.name} (uid={ctx.uid})')
    print(f'Params: p1={p1}, p2={p2}')
    print('accesskey = {}'.format(ctx.get_secret('ACCESS_KEY')))
    print('file\n{}\n'.format(ctx.input_artifact('infile.txt').get()))

    ctx.log_output('accuracy', p1 * 2)
    for i in range(1,4):
        ctx.log_metric('loss', 2*i, i)
    ctx.log_artifact('chart', 'chart.png')


if __name__ == "__main__":
    ex = get_or_create_ctx('mytask')
    my_func(ex)
    print(ex.to_yaml())
```

### Replacing Runtime Context Parameters form CLI

`python -m mlrun run -p p1=5 -s secrets.txt -i infile.txt=s3://mybucket/infile.txt test2.py`

when running the command above:
* the parameter `p1` will be overwritten with `5`
* the file `infile.txt` will be loaded from a remote S3 bucket
* credentials (for S3 and the app) will be loaded from the `secrets.txt` file