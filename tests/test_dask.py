from pprint import pprint

from conftest import rundb_path, tag_test
from mlrun import new_function, NewRun
import pytest


has_dask = False
try:
    import dask  # noqa
    has_dask = True
except ImportError:
    pass


def my_func(context, p1=1, p2='a-string'):
    print(f'Run: {context.name} (uid={context.uid})')
    print(f'Params: p1={p1}, p2={p2}\n')

    context.log_result('accuracy', p1 * 2)
    context.log_metric('loss', 7)
    context.log_artifact('chart', body='abc')
    return 'tst-me-{}'.format(context.iteration)


@pytest.mark.skipif(not has_dask, reason='missing dask')
def test_dask_local():
    spec = tag_test(NewRun(params={'p1': 3, 'p2': 'vv'}), 'test_dask_local')
    run = new_function(command='dask://', rundb=rundb_path).run(
        spec, handler=my_func)
    pprint(run.to_dict())


@pytest.mark.skipif(not has_dask, reason='missing dask')
def test_dask_local_hyper():
    task = NewRun().with_hyper_params({'p1': [5, 2, 3]}, 'max.accuracy')
    spec = tag_test(task, 'test_dask_local_hyper')
    run = new_function(command='dask://').run(spec, handler=my_func)
    pprint(run.to_dict())
