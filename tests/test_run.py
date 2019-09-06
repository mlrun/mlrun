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

from os import environ

import pytest

from conftest import (examples_path, has_secrets, here, out_path, rundb_path,
                      tag_test)
from mlrun import new_runner, NewRun, RunObject
from mlrun.utils import run_keys, update_in


def my_func(context, p1=1, p2='a-string'):
    print(f'Run: {context.name} (uid={context.uid})')
    print(f'Params: p1={p1}, p2={p2}\n')
    print('file\n{}\n'.format(context.get_input('infile.txt').get()))

    context.log_result('accuracy', p1 * 2)
    context.log_metric('loss', 7)
    context.log_artifact('chart', body='abc')


def verify_state(result: RunObject):
    state = result.status.state
    assert state == 'completed', 'wrong state ({}) {}'.format(state, result.status.error)


base_spec = NewRun(params={'p1':8}, out_path=out_path)
base_spec.spec.inputs = {'infile.txt': 'infile.txt'}

s3_spec = base_spec.copy().with_secrets('file', 'secrets.txt')
s3_spec.spec.inputs = {'infile.txt': 's3://yarons-tests/infile.txt'}


def test_noparams():
    environ['MLRUN_META_DBPATH'] = rundb_path
    result = new_runner().run(handler=my_func)

    assert result.output('accuracy') == 2, 'failed to run'
    assert result.status.output_artifacts[0].get('key') == 'chart', 'failed to run'


def test_with_params():
    spec = tag_test(base_spec, 'test_with_params')
    result = new_runner(rundb=rundb_path).run(spec, handler=my_func)

    assert result.output('accuracy') == 16, 'failed to run'
    assert result.status.output_artifacts[0].get('key') == 'chart', 'failed to run'


@pytest.mark.skipif(not has_secrets(), reason='no secrets')
def test_with_params_s3():
    spec = tag_test(s3_spec, 'test_with_params')
    result = new_runner(rundb=rundb_path).run(spec, handler=my_func)

    assert result.output('accuracy') == 16, 'failed to run'
    assert result.status.output_artifacts[0].get('key') == 'chart', 'failed to run'


def test_handler_project():
    spec = tag_test(base_spec, 'test_handler_project')
    spec.metadata.project = 'myproj'
    spec.metadata.labels = {'owner': 'yaronh'}
    result = new_runner(rundb=rundb_path).run(spec, handler=my_func)
    print(result)
    assert result.output('accuracy') == 16, 'failed to run'
    verify_state(result)


def test_handler_hyper():
    run_spec = tag_test(base_spec, 'test_handler_hyper')
    run_spec.with_hyper_params({'p1': [1, 5, 3]}, selector='max.accuracy')
    result = new_runner(rundb=rundb_path).run(run_spec, handler=my_func)
    print(result)
    assert len(result.status.iterations) == 3+1, 'hyper parameters test failed'
    assert result.status.outputs['best_iteration'] == 2, 'failed to select best iteration'
    verify_state(result)


def test_handler_hyperlist():
    run_spec = tag_test(base_spec, 'test_handler_hyperlist')
    run_spec.spec.param_file = '{}/param_file.csv'.format(here)
    result = new_runner(rundb=rundb_path).run(run_spec, handler=my_func)
    print(result)
    assert len(result.status.iterations) == 3+1, 'hyper parameters test failed'
    verify_state(result)


def test_local_runtime():
    spec = tag_test(base_spec, 'test_local_runtime')
    result = new_runner(command='{}/training.py'.format(examples_path),
                       rundb=rundb_path).run(spec)
    verify_state(result)


def test_local_handler():
    spec = tag_test(base_spec, 'test_local_runtime')
    result = new_runner(command='{}/handler.py'.format(examples_path),
                        rundb=rundb_path).run(spec, handler='my_func')
    verify_state(result)


def test_local_no_context():
    spec = tag_test(base_spec, 'test_local_no_context')
    result = new_runner(command='{}/no_ctx.py'.format(here),
                        rundb=rundb_path).run(spec, mode='noctx')
    verify_state(result)
