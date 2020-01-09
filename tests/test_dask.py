# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pprint import pprint

import pytest

from conftest import rundb_path, tag_test, verify_state
from mlrun import NewTask, new_function

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
    spec = tag_test(NewTask(params={'p1': 3, 'p2': 'vv'}), 'test_dask_local')
    run = new_function(kind='dask').run(
        spec, handler=my_func)
    verify_state(run)


@pytest.mark.skipif(not has_dask, reason='missing dask')
def test_dask_local_hyper():
    task = NewTask().with_hyper_params({'p1': [5, 2, 3]}, 'max.accuracy')
    spec = tag_test(task, 'test_dask_local_hyper')
    run = new_function(kind='dask').run(spec, handler=my_func)
    verify_state(run)
    assert len(run.status.iterations) == 3+1, 'hyper parameters test failed'
    pprint(run.to_dict())
