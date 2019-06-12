from mlrun.runtimes import get_or_create_ctx

def my_func(ctx):
    p1 = ctx.get_param('p1', 1)
    p2 = ctx.get_param('p2', 'a-string')

    print(f'Run: {ctx.name} (uid={ctx.uid})')
    print(f'Params: p1={p1}, p2={p2}\n')
    print('file\n{}\n'.format(ctx.input_artifact('infile.txt').get()))

    ctx.log_output('accuracy', p1 * 2)
    ctx.log_metric('loss', 7)
    ctx.log_artifact('chart.png')


def test_noparams():
    ex = get_or_create_ctx('mytask')
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

