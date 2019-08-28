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

import json
from copy import deepcopy
from os import listdir
from tempfile import mktemp

import pytest
import yaml

from conftest import has_secrets, out_path, rundb_path
from mlrun.artifacts import ChartArtifact, TableArtifact
from mlrun.run import get_or_create_ctx, run_start
from mlrun.utils import run_keys

run_spec =  {'metadata':
                 {'labels': {'owner': 'yaronh', 'tests': 'kfp'}},
             'spec':
                 {'parameters': {'p1': 5},
                  'input_objects': [],
                  run_keys.output_path: out_path,
                  'secret_sources': [{'kind': 'file', 'source': 'secrets.txt'}]}}


def my_job(struct):
    # load MLRUN runtime context (will be set by the runtime framework e.g. KubeFlow)
    context = get_or_create_ctx('kfp_test', spec=struct)

    # get parameters from the runtime context (or use defaults)
    p1 = context.get_param('p1', 1)
    p2 = context.get_param('p2', 'a-string')

    # access input metadata, values, files, and secrets (passwords)
    print(f'Run: {context.name} (uid={context.uid})')
    print(f'Params: p1={p1}, p2={p2}')
    print('accesskey = {}'.format(context.get_secret('ACCESS_KEY')))
    print('file\n{}\n'.format(context.get_object('infile.txt').get()))

    # RUN some useful code e.g. ML training, data prep, etc.

    # log scalar result values (job result metrics)
    context.log_result('accuracy', p1 * 2)
    context.log_result('loss', p1 * 3)

    # log various types of artifacts (file, web page, table), will be versioned and visible in the UI
    context.log_artifact('model.txt', body=b'abc is 123')
    context.log_artifact('results.html', body=b'<b> Some HTML <b>', viewer='web-app')
    context.log_artifact(TableArtifact('dataset.csv', '1,2,3\n4,5,6\n',
                                       viewer='table', header=['A', 'B', 'C']))

    # create a chart output (will show in the pipelines UI)
    chart = ChartArtifact('chart.html')
    chart.header = ['Epoch', 'Accuracy', 'Loss']
    for i in range(1, 8):
        chart.add_row([i, i / 20 + 0.75, 0.30 - i / 20])
    context.log_artifact(chart)
    return context


@pytest.mark.skipif(not has_secrets(), reason='no secrets')
def test_kfp_run():
    tmpdir = mktemp()
    spec = deepcopy(run_spec)
    spec['spec'][run_keys.output_path] = tmpdir
    print(tmpdir)
    result = run_start(spec, handler=my_job, rundb=rundb_path, kfp=True)
    print(result['status']['output_artifacts'])
    alist = listdir(tmpdir)
    expected = ['chart.html', 'dataset.csv', 'model.txt', 'results.html']
    for a in expected:
        assert a in alist, f'artifact {a} was not generated'
    assert result['status']['outputs'].get('accuracy') == 10, 'failed to run'


@pytest.mark.skipif(not has_secrets(), reason='no secrets')
def test_kfp_hyper():
    tmpdir = mktemp()
    spec = deepcopy(run_spec)
    spec['spec'][run_keys.output_path] = tmpdir
    spec['spec']['hyperparams'] = {'p1': [1, 2, 3]}
    print(tmpdir)
    result = run_start(spec, handler=my_job, rundb=rundb_path,
                       kfp=True)
    alist = listdir(tmpdir)
    print(alist)
    print(listdir('/tmp'))
    with open('/tmp/iterations') as fp:
        iter = json.load(fp)
        print(yaml.dump(iter))
    assert len(iter) == 3+1, 'didnt see expected iterations file output'
