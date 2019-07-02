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

def my_func(ctx):
    p1 = ctx.get_param('p1', 1)
    p2 = ctx.get_param('p2', 'a-string')

    print(f'Run: {ctx.name} (uid={ctx.uid})')
    print(f'Params: p1={p1}, p2={p2}\n')
    print('file\n{}\n'.format(ctx.get_object('infile.txt').get()))

    ctx.log_output('accuracy', p1 * 2)
    ctx.log_metric('loss', 7)
    ctx.log_artifact('chart')


def test_noparams():
    ex = get_or_create_ctx('mytask', rundb='./')
    my_func(ex)

    result = ex.to_dict()
    assert result['status']['outputs'].get('accuracy') == 2, 'failed to run'
    assert result['status']['output_artifacts'][0].get('key') == 'chart', 'failed to run'


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
    assert result['status']['output_artifacts'][0].get('key') == 'chart', 'failed to run'

run_spec =  {'metadata':
                 {'labels': {'runtime': 'local', 'owner': 'yaronh'}},
             'spec':
                 {'parameters': {'p1': 5}, 'input_objects': [], 'secret_sources': [{'kind': 'file', 'source': 'secrets.txt'}]}}


def test_runtime():
    print(run_start(run_spec, 'example1.py', rundb='./'))