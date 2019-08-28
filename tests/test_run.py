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
from mlrun.run import get_or_create_ctx, run_start
from mlrun.utils import run_keys, update_in


def my_func(spec=None):
    ctx = get_or_create_ctx('run_test', spec=spec)
    p1 = ctx.get_param('p1', 1)
    p2 = ctx.get_param('p2', 'a-string')

    print(f'Run: {ctx.name} (uid={ctx.uid})')
    print(f'Params: p1={p1}, p2={p2}\n')
    print('file\n{}\n'.format(ctx.get_object('infile.txt').get()))

    ctx.log_result('accuracy', p1 * 2)
    ctx.log_metric('loss', 7)
    ctx.log_artifact('chart')
    return ctx


def test_noparams():
    environ['MLRUN_META_DBPATH'] = rundb_path
    result = my_func().to_dict()

    assert result['status']['outputs'].get('accuracy') == 2, 'failed to run'
    assert result['status'][run_keys.output_artifacts][0].get('key') == 'chart', 'failed to run'


basespec = { 'metadata': {}, 'spec': {
    'parameters':{'p1':8},
    'secret_sources': [{'kind':'file', 'source': 'secrets.txt'}],
    run_keys.output_path: out_path,
    run_keys.input_objects: [{'key':'infile.txt', 'path':'s3://yarons-tests/infile.txt'}],
}}


@pytest.mark.skipif(not has_secrets(), reason='no secrets')
def test_with_params():
    environ['MLRUN_META_DBPATH'] = rundb_path
    spec = tag_test(basespec, 'test_with_params')

    result = my_func(spec).to_dict()
    assert result['status']['outputs'].get('accuracy') == 16, 'failed to run'
    assert result['status'][run_keys.output_artifacts][0].get('key') == 'chart', 'failed to run'

basespec2 =  {'metadata':
                 {},
             'spec':
                 {'parameters': {'p1': 5},
                  run_keys.input_objects: [],
                  run_keys.output_path: out_path,
                  'secret_sources': [
                      {'kind': 'file', 'source': 'secrets.txt'}]}}


basespec_project =  {'metadata':
                 {'labels': {'owner': 'yaronh'},
                  'project': 'myproj'},
             'spec':
                 {'parameters': {'p1': 5},
                  'input_objects': [],
                  run_keys.output_path: out_path,
                  'secret_sources': [{'kind': 'file', 'source': 'secrets.txt'}]}}


def verify_state(result):
    state = result['status']['state']
    assert state == 'completed', f'wrong state ({state}) ' + result['status'].get('error', '')


@pytest.mark.skipif(not has_secrets(), reason='no secrets')
def test_handler():
    run_spec = tag_test(basespec2, 'test_handler')
    result = run_start(run_spec, handler=my_func, rundb=rundb_path)
    print(result)
    assert result['status']['outputs'].get('accuracy') == 10, 'failed to run'
    verify_state(result)


@pytest.mark.skipif(not has_secrets(), reason='no secrets')
def test_handler_project():
    run_spec_project = tag_test(basespec_project, 'test_handler_project')
    result = run_start(run_spec_project, handler=my_func, rundb=rundb_path)
    print(result)
    assert result['status']['outputs'].get('accuracy') == 10, 'failed to run'
    verify_state(result)


@pytest.mark.skipif(not has_secrets(), reason='no secrets')
def test_handler_empty_hyper():
    run_spec = tag_test(basespec2, 'test_handler_empty_hyper')
    run_spec['spec']['hyperparams'] = {'p1': [2, 4]}
    result = run_start(run_spec, handler=my_func, rundb=rundb_path)
    verify_state(result)


@pytest.mark.skipif(not has_secrets(), reason='no secrets')
def test_handler_hyper():
    run_spec = tag_test(basespec2, 'test_handler_hyper')
    run_spec['spec']['hyperparams'] = {'p1': [1, 2, 3]}
    result = run_start(run_spec, handler=my_func, rundb=rundb_path)
    print(result)
    assert len(result['status']['iterations']) == 3+1, 'hyper parameters test failed'
    verify_state(result)


@pytest.mark.skipif(not has_secrets(), reason='no secrets')
def test_handler_hyperlist():
    run_spec = tag_test(basespec2, 'test_handler_hyperlist')
    run_spec['spec']['param_file'] = 'param_file.csv'
    result = run_start(run_spec, handler=my_func, rundb=rundb_path)
    print(result)
    assert len(result['status']['iterations']) == 3+1, 'hyper parameters test failed'
    verify_state(result)


@pytest.mark.skipif(not has_secrets(), reason='no secrets')
def test_local_runtime():
    spec = tag_test(basespec, 'test_local_runtime')
    result = run_start(spec, command=f'{examples_path}/training.py', rundb=rundb_path)
    verify_state(result)


@pytest.mark.skipif(not has_secrets(), reason='no secrets')
def test_local_no_context():
    spec = tag_test(basespec, 'test_local_no_context')
    result = run_start(spec, command=f'{here}/no_ctx.py', rundb=rundb_path, mode='noctx')
    verify_state(result)
