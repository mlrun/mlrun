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

from mlrun.runtimes import get_or_create_ctx, run_start
from os import environ


def my_func(spec=None):
    ctx = get_or_create_ctx('mytask', spec=spec)
    p1 = ctx.get_param('p1', 1)
    p2 = ctx.get_param('p2', 'a-string')

    print(f'Run: {ctx.name} (uid={ctx.uid})')
    print(f'Params: p1={p1}, p2={p2}\n')
    print('file\n{}\n'.format(ctx.get_object('infile.txt').get()))

    ctx.log_output('accuracy', p1 * 2)
    ctx.log_metric('loss', 7)
    ctx.log_artifact('chart')
    return ctx


def test_noparams():
    environ['MLRUN_META_DBPATH'] = './'
    result = my_func().to_dict()

    assert result['status']['outputs'].get('accuracy') == 2, 'failed to run'
    assert result['status']['output_artifacts'][0].get('key') == 'chart', 'failed to run'


spec = {'spec': {
    'parameters':{'p1':8},
    'secret_sources': [{'kind':'file', 'source': 'secrets.txt'}],
    'input_artifacts': [{'key':'infile.txt', 'path':'s3://yarons-tests/infile.txt'}],
}}


def test_with_params():
    environ['MLRUN_META_DBPATH'] = './'
    result = my_func(spec).to_dict()
    assert result['status']['outputs'].get('accuracy') == 16, 'failed to run'
    assert result['status']['output_artifacts'][0].get('key') == 'chart', 'failed to run'

run_spec =  {'metadata':
                 {'labels': {'owner': 'yaronh'}},
             'spec':
                 {'parameters': {'p1': 5},
                  'input_objects': [],
                  'secret_sources': [{'kind': 'file', 'source': 'secrets.txt'}]}}


run_spec_project =  {'metadata':
                 {'labels': {'owner': 'yaronh'},
                  'project': 'myproj'},
             'spec':
                 {'parameters': {'p1': 5},
                  'input_objects': [],
                  'secret_sources': [{'kind': 'file', 'source': 'secrets.txt'}]}}


def test_handler():
    result = run_start(run_spec, handler=my_func, rundb='./')
    print(result)
    assert result['status']['outputs'].get('accuracy') == 10, 'failed to run'


def test_handler_project():
    result = run_start(run_spec_project, handler=my_func, rundb='./')
    print(result)
    assert result['status']['outputs'].get('accuracy') == 10, 'failed to run'


def test_handler_hyper():
    result = run_start(run_spec, handler=my_func, rundb='./',
                       hyperparams={'p1': [1, 2, 3]})
    print(result)
    assert len(result['status']['iterations']) == 3, 'hyper parameters test failed'


def test_local_runtime():
    print(run_start(run_spec, command='example1.py', rundb='./'))