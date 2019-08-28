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

import time
from threading import Thread

import pytest

from conftest import has_secrets, out_path, rundb_path, tag_test
from http_srv import create_function
from mlrun import get_or_create_ctx, run_start
from mlrun.utils import run_keys


def myfunction(context, event):
    ctx = get_or_create_ctx('nuclio-test', event=event)
    p1 = ctx.get_param('p1', 1)
    p2 = ctx.get_param('p2', 'a-string')

    context.logger.info(
        f'Run: {ctx.name} uid={ctx.uid}:{ctx.iteration} Params: p1={p1}, p2={p2}')

    time.sleep(1)

    # log scalar values (KFP metrics)
    ctx.log_result('accuracy', p1 * 2)
    ctx.log_result('latency', p1 * 3)

    # log various types of artifacts (and set UI viewers)
    ctx.log_artifact('test.txt', body=b'abc is 123')
    ctx.log_artifact('test.html', body=b'<b> Some HTML <b>', viewer='web-app')

    context.logger.info('run complete!')
    return ctx.to_json()


basespec = {'spec': {
    'parameters':{'p1':8},
    'secret_sources': [{'kind':'file', 'source': 'secrets.txt'}],
    run_keys.output_path: out_path,
    run_keys.input_objects: [{'key':'infile.txt', 'path':'s3://yarons-tests/infile.txt'}],
}}


def verify_state(result):
    state = result['status']['state']
    assert state == 'completed', f'wrong state ({state}) ' + result['status'].get('error', '')


@pytest.mark.skipif(not has_secrets(), reason='no secrets')
def test_simple_function():
    Thread(target=create_function, args=(myfunction, 4444)).start()
    time.sleep(2)

    spec = tag_test(basespec, 'simple_function')
    result = run_start(spec, command='http://localhost:4444',
                       rundb=rundb_path)
    print(result)
    verify_state(result)


@pytest.mark.skipif(not has_secrets(), reason='no secrets')
def test_hyper_function():
    Thread(target=create_function, args=(myfunction, 4444))
    time.sleep(2)

    spec = tag_test(basespec, 'hyper_function')
    spec['spec']['hyperparams'] = {'p1': [1, 2, 3]}
    result = run_start(spec, command='http://localhost:4444',
                       rundb=rundb_path)
    print(result)
    verify_state(result)
