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

import csv
from os import listdir
from tempfile import mktemp
import pandas as pd

import yaml

from tests.conftest import out_path
from mlrun.artifacts import ChartArtifact, TableArtifact
from mlrun import NewTask, new_function


run_spec = NewTask(
    params={'p1': 5},
    out_path=out_path,
    outputs=['model.txt', 'chart.html', 'iteration_results'],
).set_label('tests', 'kfp')


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

    # log various types of artifacts (file, web page, table), will be
    # versioned and visible in the UI
    context.log_artifact('model', body=b'abc is 123', local_path='model.txt')
    context.log_artifact(
        'results', local_path='results.html', body=b'<b> Some HTML <b>'
    )
    context.log_artifact(
        TableArtifact(
            'dataset',
            '1,2,3\n4,5,6\n',
            format='csv',
            viewer='table',
            header=['A', 'B', 'C'],
        )
    )

    # create a chart output (will show in the pipelines UI)
    chart = ChartArtifact('chart')
    chart.header = ['Epoch', 'Accuracy', 'Loss']
    for i in range(1, 8):
        chart.add_row([i, i / 20 + 0.75, 0.30 - i / 20])
    context.log_artifact(chart)

    raw_data = {
        'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
        'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
        'age': [42, 52, 36, 24, 73],
        'postTestScore': [25, 94, 57, 62, 70],
    }
    df = pd.DataFrame(
        raw_data, columns=['first_name', 'last_name', 'age', 'postTestScore']
    )
    context.log_dataset('mydf', df=df)


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
    assert result.status.state == 'completed', 'wrong state ({}) {}'.format(
        result.status.state, result.status.error
    )


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
    res_file = tmpdir + '/' + 'iteration_results.csv'
    with open(res_file) as fp:
        count = 0
        for row in csv.DictReader(fp):
            print(yaml.dump(row))
            count += 1
    assert count == 3, 'didnt see expected iterations file output'
    assert result.status.state == 'completed', 'wrong state ({}) {}'.format(
        result.status.state, result.status.error
    )
