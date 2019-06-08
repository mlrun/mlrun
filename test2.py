import json
import os
from mlrun.runtimes import LocalRuntime


def my_func(ctx):
    p1 = ctx.get_or_set_param('p1', 1)
    p2 = ctx.get_or_set_param('p2', 'a-string')

    print(f'Run: {ctx.name} (uid={ctx.uid})')
    print(f'Params: p1={p1}, p2={p2}')
    print('accesskey = {}'.format(ctx.get_secret('ACCESS_KEY')))
    print('file\n{}\n'.format(ctx.input_artifact('infile.txt').get()))

    ctx.log_output('accuracy', p1 * 2)
    for i in range(1,4):
        ctx.log_metric('loss', 2*i, i)
    ctx.log_artifact('chart', 'chart.png')


if __name__ == "__main__":
    ex = LocalRuntime('mytask')
    my_func(ex)
    print(ex.to_yaml())