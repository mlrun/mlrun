from pprint import pprint

import pytest

from conftest import tag_test, verify_state
from mlrun import NewTask, new_function
from tempfile import mkdtemp

log_config = '''
logging:
  distributed: error
  distributed.client: error
  bokeh: error
  tornado: error
'''

has_dask = False
try:
    import dask  # noqa
    has_dask = True
    tmp_dir = mkdtemp(prefix='mlrun-dask-config')
    with open(f'{tmp_dir}/logging.yml', 'w') as out:
        out.write(log_config)
    dask.config.paths.append(tmp_dir)
    dask.config.refresh()
except ImportError:
    pass

try:
    from dask.distributed import default_client

    @pytest.fixture
    def dask_client():
        yield
        default_client().close()

except ImportError:
    @pytest.fixture
    def dask_client():
        yield


def my_func(context, p1=1, p2='a-string'):
    print(f'Run: {context.name} (uid={context.uid})')
    print(f'Params: p1={p1}, p2={p2}\n')

    context.log_result('accuracy', p1 * 2)
    context.log_metric('loss', 7)
    context.log_artifact('chart', body='abc')
    return 'tst-me-{}'.format(context.iteration)


@pytest.mark.skipif(not has_dask, reason='missing dask')
def test_dask_local(dask_client):
    spec = tag_test(NewTask(params={'p1': 3, 'p2': 'vv'}), 'test_dask_local')
    run = new_function(kind='dask').run(
        spec, handler=my_func)
    verify_state(run)


@pytest.mark.skipif(not has_dask, reason='missing dask')
def test_dask_local_hyper(dask_client):
    task = NewTask().with_hyper_params({'p1': [5, 2, 3]}, 'max.accuracy')
    spec = tag_test(task, 'test_dask_local_hyper')
    run = new_function(kind='dask').run(spec, handler=my_func)
    verify_state(run)
    assert len(run.status.iterations) == 3+1, 'hyper parameters test failed'
    pprint(run.to_dict())
