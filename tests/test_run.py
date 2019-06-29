from mlrun.runtimes import get_or_create_ctx, run_start

def my_func(ctx):
    p1 = ctx.get_param('p1', 1)
    p2 = ctx.get_param('p2', 'a-string')

    print(f'Run: {ctx.name} (uid={ctx.uid})')
    print(f'Params: p1={p1}, p2={p2}\n')
    print('file\n{}\n'.format(ctx.get_object('infile.txt').get()))

    ctx.log_output('accuracy', p1 * 2)
    ctx.log_metric('loss', 7)
    ctx.log_artifact('chart.png')


def test_noparams():
    ex = get_or_create_ctx('mytask', rundb='./')
    my_func(ex)

    result = ex.to_dict()
    assert result['status']['outputs'].get('accuracy') == 2, 'failed to run'
    assert result['status']['output_artifacts'][0].get('key') == 'chart.png', 'failed to run'


spec = {'spec': {
    'parameters':{'p1':8},
    'secret_sources': [{'kind':'file', 'source': 'secrets.txt'}],
    'input_artifacts': [{'key':'infile.txt', 'path':'s3://yarons-tests/infile.txt'}],
}}


def test_with_params():
    ex = get_or_create_ctx('task2', spec=spec)
    my_func(ex)

    result = ex.to_dict()
    assert result['status']['outputs'].get('accuracy') == 16, 'failed to run'
    assert result['status']['output_artifacts'][0].get('key') == 'chart.png', 'failed to run'

run_spec =  {'metadata':
                 {'labels': {'runtime': 'local', 'owner': 'yaronh'}},
             'spec':
                 {'parameters': {'p1': 5}, 'input_objects': [], 'secret_sources': [{'kind': 'file', 'source': 'secrets.txt'}]}}


def test_runtime():
    print(run_start(run_spec, 'example1.py', save_to='./'))