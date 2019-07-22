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
import time
from os import path, environ
from urllib.parse import urlparse
import yaml
import pathlib
import pandas as pd

from .utils import get_in, match_labels, dict_to_yaml, flatten
from .datastore import StoreManager
from .render import run_to_html, runs_to_html, artifacts_to_html


def get_run_db(url=''):
    if not url:
        url = environ.get('MLRUN_META_DBPATH', './')

    p = urlparse(url)
    scheme = p.scheme.lower()
    if '://' not in url or scheme in ['file', 's3', 'v3io', 'v3ios']:
        db = FileRunDB(url)
    else:
        raise ValueError('unsupported run DB scheme ({})'.format(scheme))
    return db


class RunList(list):

    def to_rows(self):
        rows = []
        head = ['uid', 'iter', 'start', 'state', 'name', 'labels',
                'inputs', 'parameters', 'results', 'artifacts']
        for run in self:
            row = [
                get_in(run, 'metadata.uid', ''),
                get_in(run, 'metadata.iteration', ''),
                get_in(run, 'status.start_time', ''),
                get_in(run, 'status.state', ''),
                get_in(run, 'metadata.name', ''),
                get_in(run, 'metadata.labels', ''),
                get_in(run, 'spec.input_objects', ''),
                get_in(run, 'spec.parameters', ''),
                get_in(run, 'status.outputs', ''),
                get_in(run, 'status.output_artifacts', []),
            ]
            rows.append(row)

        return [head] + rows

    def to_df(self, flat=False):
        rows = self.to_rows()
        df = pd.DataFrame(rows[1:], columns=rows[0]) #.set_index('iter')
        df['start'] = pd.to_datetime(df['start'])

        if flat:
            df = flatten(df, 'labels')
            df = flatten(df, 'parameters', 'param_')
            df = flatten(df, 'outputs', 'out_')

        return df

    def show(self, display=True):
        html = runs_to_html(self.to_df(), display)
        if not display:
            return html


class ArtifactList(list):

    def to_rows(self):
        rows = []
        head = {'key': '', 'kind': '', 'path': 'target_path', 'hash': '',
                'viewer': '', 'updated': '', 'description': '', 'producer': '',
                'sources': '', 'labels': ''}
        for artifact in self:
            row = [get_in(artifact, v or k, '') for k, v in head.items()]
            rows.append(row)

        return [head.keys()] + rows

    def to_df(self, flat=False):
        rows = self.to_rows()
        df = pd.DataFrame(rows[1:], columns=rows[0])
        df['updated'] = pd.to_datetime(df['updated'], unit='s')

        if flat:
            df = flatten(df, 'producer', 'prod_')
            df = flatten(df, 'sources', 'src_')

        return df

    def show(self, display=True):
        html = artifacts_to_html(self.to_df(), display)
        if not display:
            return html


class RunDBInterface:
    kind = ''

    def connect(self, secrets=None):
        return self

    def store_run(self, struct, uid, project='', commit=False):
        pass

    def read_run(self, uid, project=''):
        pass

    def list_runs(self, name='', project='', labels=[], sort=False):
        pass

    def store_artifact(self, key, artifact, uid, tag='', project=''):
        pass

    def read_artifact(self, key, tag='', project=''):
        pass

    def list_artifacts(self, name='', project='', tag='', labels=[]):
        pass

    def store_metric(self, keyvals={}, timestamp=None, labels={}):
        pass

    def read_metric(self, keys, query=''):
        pass


class FileRunDB(RunDBInterface):
    kind = 'file'

    def __init__(self, dirpath='', format='.yaml'):
        self.format = format
        self.dirpath = dirpath
        self._datastore = None
        self._subpath = None

    def connect(self, secrets=None):
        sm = StoreManager(secrets)
        self._datastore, self._subpath = sm.get_or_create_store(self.dirpath)
        return self

    def store_run(self, struct, uid, project='', commit=False):
        if self.format == '.yaml':
            data = dict_to_yaml(struct)
        else:
            data = json.dumps(struct)
        filepath = self._filepath('runs', project, uid, '') + self.format
        self._datastore.put(filepath, data)

    def read_run(self, uid, project='', display=True):
        filepath = self._filepath('runs', project, uid, '') + self.format
        data = self._datastore.get(filepath)
        result = self._loads(data)

        run_to_html(result, display)

        return result

    def list_runs(self, name='', project='', labels=[], sort=False):
        filepath = self._filepath('runs', project)
        results = RunList()
        if isinstance(labels, str):
            labels = labels.split(',')
        for run in self._load_list(filepath):
            if (name == '' or name in get_in(run, 'metadata.name', ''))\
                    and match_labels(get_in(run, 'metadata.labels', {}), labels):
                results.append(run)

        if sort:
            results.sort(key = lambda i: get_in(
                i, ['status','start_time']), reverse=True)

        return results

    def store_artifact(self, key, artifact, uid, tag='', project=''):
        artifact.updated = time.time()
        if self.format == '.yaml':
            data = artifact.to_yaml()
        else:
            data = artifact.to_json()
        filepath = self._filepath('artifacts', project, key, uid) + self.format
        self._datastore.put(filepath, data)
        filepath = self._filepath('artifacts', project, key, tag or 'latest') + self.format
        self._datastore.put(filepath, data)

    def read_artifact(self, key, tag='', project=''):
        filepath = self._filepath('artifacts', project, key, tag) + self.format
        data = self._datastore.get(filepath)
        return self._loads(data)

    def list_artifacts(self, name='', project='', tag='', labels=[]):
        filepath = self._filepath('artifacts', project, tag=tag or 'latest')
        results = ArtifactList()
        if isinstance(labels, str):
            labels = labels.split(',')
        for artifact in self._load_list(filepath):
            if (name == '' or name in get_in(artifact, 'key', ''))\
                    and match_labels(get_in(artifact, 'labels', {}), labels):
                results.append(artifact)

        return results

    def _filepath(self, table, project, key='', tag=''):
        if tag:
            key = '/' + key
        if project:
            return path.join(self.dirpath, '{}/{}/{}{}'.format(table, project, tag, key))
        else:
            return path.join(self.dirpath, '{}/{}{}'.format(table, tag, key))

    def _dumps(self, obj):
        if self.format == '.yaml':
            return obj.to_yaml()
        else:
            return obj.to_json()

    def _loads(self, data):
        if self.format == '.yaml':
            return yaml.load(data, Loader=yaml.FullLoader)
        else:
            return json.loads(data)

    def _load_list(self, dirpath):
        for p in pathlib.Path(dirpath).iterdir():
            if p.is_file() and p.suffix == self.format:
                yield self._loads(p.read_text())


