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
from os import listdir
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

import mlrun
import mlrun.errors
from tests.conftest import rundb_path

# fixtures for test, aren't used directly so we need to ignore the lint here
from tests.common_fixtures import patch_file_forbidden  # noqa: F401

mlrun.mlconf.dbpath = rundb_path

raw_data = {
    'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
    'age': [42, 52, 36, 24, 73],
}
df = pd.DataFrame(raw_data, columns=['name', 'age'])


def test_in_memory():
    context = mlrun.get_or_create_ctx('test-in-mem')
    context.artifact_path = 'memory://'
    context.log_artifact('k1', body='abc')
    context.log_dataset('k2', df=df)

    data = mlrun.datastore.set_in_memory_item('aa', '123')
    in_memory_store = mlrun.datastore.get_in_memory_items()
    new_df = mlrun.run.get_dataitem("memory://k2").as_df()

    assert len(in_memory_store) == 3, 'data not written properly to in mem store'
    assert data.get() == '123', 'in mem store failed to get/put'
    assert len(new_df) == 5, 'in mem store failed dataframe test'
    assert (
        mlrun.run.get_dataitem("memory://k1").get() == 'abc'
    ), 'failed to log in mem artifact'


def test_file():
    with TemporaryDirectory() as tmpdir:
        print(tmpdir)

        data = mlrun.run.get_dataitem(tmpdir + '/test1.txt')
        data.put('abc')
        assert data.get() == b'abc', 'failed put/get test'
        assert data.stat().size == 3, 'got wrong file size'
        print(data.stat())

        context = mlrun.get_or_create_ctx('test-file')
        context.artifact_path = tmpdir
        k1 = context.log_artifact('k1', body='abc', local_path='x.txt')
        k2 = context.log_dataset('k2', df=df, format='csv', db_key='k2key')
        print('k2 url:', k2.get_store_url())

        alist = listdir(tmpdir)
        print(alist)
        assert mlrun.run.get_dataitem(tmpdir).listdir() == alist, 'failed listdir'

        expected = ['test1.txt', 'x.txt', 'k2.csv']
        for a in expected:
            assert a in alist, 'artifact {} was not generated'.format(a)

        new_fd = mlrun.run.get_dataitem(tmpdir + '/k2.csv').as_df()

        assert len(new_fd) == 5, 'failed dataframe test'
        assert (
            mlrun.run.get_dataitem(tmpdir + '/x.txt').get() == b'abc'
        ), 'failed to log in file artifact'

        name = k2.get_store_url()
        artifact, _ = mlrun.artifacts.get_artifact_meta(name)
        print(artifact.to_yaml())
        mlrun.artifacts.update_dataset_meta(
            artifact, extra_data={'k1': k1}, column_metadata={'age': 'great'}
        )
        artifact, _ = mlrun.artifacts.get_artifact_meta(name)
        print(artifact.to_yaml())
        assert artifact.column_metadata == {
            'age': 'great'
        }, 'failed artifact update test'


def test_parse_url_preserve_case():
    url = 'store://Hedi/mlrun-dbd7ef-training_mymodel#a5dc8e34a46240bb9a07cd9deb3609c7'
    expected_endpoint = 'Hedi'
    _, endpoint, _ = mlrun.datastore.datastore.parse_url(url)
    assert expected_endpoint, endpoint


@pytest.mark.usefixtures("patch_file_forbidden")
def test_forbidden_file_access():
    store = mlrun.datastore.datastore.StoreManager(
        secrets={'V3IO_ACCESS_KEY': 'some-access-key'}
    )

    with pytest.raises(mlrun.errors.AccessDeniedError):
        obj = store.object('v3io://some-system/some-dir/')
        obj.listdir()

    with pytest.raises(mlrun.errors.AccessDeniedError):
        obj = store.object('v3io://some-system/some-dir/some-file')
        obj.get()

    with pytest.raises(mlrun.errors.AccessDeniedError):
        obj = store.object('v3io://some-system/some-dir/some-file')
        obj.stat()
