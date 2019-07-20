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
from os import path, environ, listdir
from urllib.parse import urlparse
import yaml
import pathlib
import pandas as pd

from .utils import is_ipython, get_in, match_labels, dict_to_yaml, flatten
from .datastore import StoreManager
from .render import ipython_ui


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


def get_run(uid, project='', rundb='', secrets=None):
    db = get_run_db(rundb).connect(secrets)
    result = db.read_run(uid, project)

    if is_ipython and result:
        ipython_ui(result)

    return result


class RunList(list):

    def to_rows(self):
        rows = []
        artifacts_list = []
        head = ['uid', 'iteration', 'start', 'name', 'labels',
                'parameters', 'outputs']
        for run in self:
            row = [
                get_in(run, 'metadata.uid', ''),
                get_in(run, 'metadata.iteration', ''),
                get_in(run, 'status.start_time', ''),
                get_in(run, 'metadata.name', ''),
                get_in(run, 'metadata.labels', ''),
                get_in(run, 'spec.parameters', ''),
                get_in(run, 'status.outputs', ''),
            ]
            rows.append(row)

            artifacts = get_in(run, 'status.output_artifacts', [])
            artifacts_list.append(artifacts)

        return [head] + rows, artifacts_list

    def to_df(self, flat=True, links=False):
        rows, artifacts_list = self.to_rows()
        df = pd.DataFrame(rows[1:], columns=rows[0]) #.set_index('iter')
        df['start'] = pd.to_datetime(df['start'])

        if flat:
            df = flatten(df, 'labels')
            df = flatten(df, 'parameters', 'param_')
            df = flatten(df, 'outputs', 'out_')

        if links:
            new_list = []
            for row in artifacts_list:
                html = []
                for item in row:
                    if item:
                        html.append('<a href="{}">{}</a>'.format(item['key'], item['target_path']))
                new_list.append(', '.join(html))
            df['artifacts'] = new_list

        return df

    def show(self):
        df = self.to_df(links=True)
        import IPython
        pd.set_option('display.max_colwidth', -1)
        IPython.display.display(
            IPython.display.HTML(df.to_html(escape=False)))


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

    def store_artifact(self, key, artifact, tag='', project=''):
        pass

    def read_artifact(self, key, tag='', project=''):
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
        filepath = self._filepath('runs', project, uid, '', self.format)
        self._datastore.put(filepath, data)

    def read_run(self, uid, project=''):
        filepath = self._filepath('runs', project, uid, '', self.format)
        data = self._datastore.get(filepath)
        return self._loads(data)

    def list_runs(self, name='', project='', labels=[], sort=False):
        filepath = self._filepath('runs', project)
        results = RunList()
        for run in self._load_list(filepath):
            if (name == '' or name in get_in(run, 'metadata.name', ''))\
                    and match_labels(get_in(run, 'metadata.labels', {}), labels):
                results.append(run)

        if sort:
            results.sort(key = lambda i: get_in(
                i, ['status','start_time']), reverse=True)

        return results

    def store_artifact(self, key, artifact, tag='', project=''):
        if self.format == '.yaml':
            data = artifact.to_yaml()
        else:
            data = artifact.to_json()
        filepath = self._filepath('artifacts', project, key, tag, self.format)
        self._datastore.put(filepath, data)

    def read_artifact(self, key, tag='', project=''):
        filepath = self._filepath('artifacts', project, key, tag, self.format)
        data = self._datastore.get(filepath)
        return self._loads(data)

    def _filepath(self, table, project, uid='', tag='', format=''):
        if tag:
            tag = '/' + tag
        if project:
            return path.join(self.dirpath, '{}/{}/{}{}{}'.format(table, project, uid, tag, format))
        else:
            return path.join(self.dirpath, '{}/{}{}{}'.format(table, uid, tag, format))

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


