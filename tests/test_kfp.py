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
from os import listdir
from tempfile import mktemp

import pytest
import yaml

from conftest import has_secrets, out_path, rundb_path
from mlrun.artifacts import ChartArtifact, TableArtifact
from mlrun import NewRun, new_function
from mlrun.utils import run_keys


run_spec = NewRun(params={'p1': 5},
                  out_path=out_path,
                  outputs=['model.txt', 'chart.html']).set_label('tests', 'kfp')


def my_job(context, p1=1, p2='a-string'):

    # access input metadata, values, files, and secrets (passwords)
    print(f'Run: {context.name} (uid={context.uid})')
    print(f'Params: p1={p1}, p2={p2}')
    print('accesskey = {}'.format(context.get_secret('ACCESS_KEY')))
    print('file\n{}\n'.format(context.get_input('infile.txt').get()))

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


def test_kfp_run():
    tmpdir = mktemp()
    spec = run_spec.copy()
    spec.spec.output_path = tmpdir
    print(tmpdir)
    result = new_function(kfp=True).run(spec, handler=my_job)
    print(result.status.artifacts)
    alist = listdir(tmpdir)
    expected = ['chart.html', 'dataset.csv', 'model.txt', 'results.html']
    for a in expected:
        assert a in alist, 'artifact {} was not generated'.format(a)
    assert result.output('accuracy') == 10, 'failed to run'
    assert result.status.state == 'completed', \
        'wrong state ({}) {}'.format(result.status.state, result.status.error)


def test_kfp_hyper():
    tmpdir = mktemp()
    spec = run_spec.copy()
    spec.spec.output_path = tmpdir
    spec.with_hyper_params({'p1': [1, 2, 3]}, selector='min.loss')
    print(tmpdir)
    result = new_function(kfp=True).run(spec, handler=my_job)
    alist = listdir(tmpdir)
    print(alist)
    print(listdir('/tmp'))
    with open('/tmp/iteration_results.csv') as fp:
        print('XXXXXXXXXXXXXXXXX\nXXXXXXXXXXXXXXXXXX')
        print(fp.read())
    with open('/tmp/iterations') as fp:
        iter = json.load(fp)
        print(yaml.dump(iter))
    assert len(iter) == 3+1, 'didnt see expected iterations file output'
    assert result.status.state == 'completed', \
        'wrong state ({}) {}'.format(result.status.state, result.status.error)
