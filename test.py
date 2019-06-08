import json
import os
from mlrun.runtimes import LocalRuntime

def my_func(ctx):
    p1 = ctx.get_or_set_param('p1', 1)
    p2 = ctx.get_or_set_param('p2', 'a-string')

    print(f'Run: {ctx.name} (uid={ctx.uid})')
    print(f'Params: p1={p1}, p2={p2}\n')

    ctx.log_output('accuracy', p1 * 2)
    ctx.log_metric('loss', 7)
    ctx.log_artifact('chart', 'chart.png')


ex = LocalRuntime('mytask')
my_func(ex)

ex = LocalRuntime('task2', parameters={'p1':8})
my_func(ex)
print(ex.to_yaml())
