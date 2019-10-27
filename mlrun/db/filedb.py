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
from os import path, remove
import yaml
import pathlib
from datetime import datetime, timedelta

from ..utils import get_in, match_labels, dict_to_yaml, update_in, dict_to_json
from ..datastore import StoreManager
from ..render import run_to_html
from .base import RunDBError, RunDBInterface
from ..lists import RunList, ArtifactList


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

    def store_log(self, uid, project='', body=None, append=True):
        filepath = self._filepath('runs', project, uid, '') + '.log'
        self._datastore.put(filepath, body, append=append)

    def get_log(self, uid, project='', offset=0):
        filepath = self._filepath('runs', project, uid, '') + '.log'
        if pathlib.Path(filepath).is_file():
            with open(filepath, 'rb') as fp:
                if offset:
                    fp.seek(offset)
                return fp.read()
        return None

    def store_run(self, struct, uid, project='', commit=False):
        if self.format == '.yaml':
            data = dict_to_yaml(struct)
        else:
            data = dict_to_json(struct)
        filepath = self._filepath('runs', project, uid, '') + self.format
        self._datastore.put(filepath, data)

    def update_run(self, updates: dict, uid, project=''):
        run = self.read_run(uid, project)
        if run and updates:
            for key, val in updates.items():
                update_in(run, key, val)
        self.store_run(run, uid, project, True)

    def read_run(self, uid, project=''):
        filepath = self._filepath('runs', project, uid, '') + self.format
        data = self._datastore.get(filepath)
        return self._loads(data)

    def list_runs(self, name='', uid=None, project='', labels=[],
                  state='', sort=True, last=30):
        filepath = self._filepath('runs', project)
        results = RunList()
        if isinstance(labels, str):
            labels = labels.split(',')
        for run, _ in self._load_list(filepath, '*'):
            if (name == '' or name in get_in(run, 'metadata.name', ''))\
                    and match_labels(get_in(run, 'metadata.labels', {}), labels)\
                    and (state == '' or get_in(run, 'status.state', '') == state)\
                    and (not uid or uid == get_in(run, 'metadata.uid', '')):
                results.append(run)

        if sort or last:
            results.sort(key=lambda i: get_in(
                i, ['status', 'start_time'], ''), reverse=True)
        if last and len(results) > last:
            return RunList(results[:last])
        return results

    def del_run(self, uid, project=''):
        filepath = self._filepath('runs', project, uid, '') + self.format
        self._safe_del(filepath)

    def del_runs(self, name='', project='', labels=[], state='', days_ago=0):
        if not name and not state and not days_ago:
            raise RunDBError('filter is too wide, select name and/or state and/or days_ago')

        filepath = self._filepath('runs', project)
        if isinstance(labels, str):
            labels = labels.split(',')

        if days_ago:
            days_ago = datetime.now() - timedelta(days=days_ago)

        def date_before(run):
            return datetime.strptime(get_in(run, 'status.start_time', ''),
                                     '%Y-%m-%d %H:%M:%S.%f') < days_ago

        for run, p in self._load_list(filepath, '*'):
            if (name == '' or name == get_in(run, 'metadata.name', ''))\
                    and match_labels(get_in(run, 'metadata.labels', {}), labels)\
                    and (state == '' or get_in(run, 'status.state', '') == state)\
                    and (not days_ago or date_before(run)):

                self._safe_del(p)

    def store_artifact(self, key, artifact, uid, tag='', project=''):
        artifact['updated'] = time.time()
        if self.format == '.yaml':
            data = dict_to_yaml(artifact)
        else:
            data = dict_to_json(artifact)
        filepath = self._filepath('artifacts', project, key, uid) + self.format
        self._datastore.put(filepath, data)
        filepath = self._filepath('artifacts', project, key, tag or 'latest') + self.format
        self._datastore.put(filepath, data)

    def read_artifact(self, key, tag='', project=''):
        filepath = self._filepath('artifacts', project, key, tag) + self.format
        data = self._datastore.get(filepath)
        return self._loads(data)

    def list_artifacts(self, name='', project='', tag='', labels=[]):
        tag = tag or 'latest'
        print(f'reading artifacts in {project} name/mask: {name} tag: {tag} ...')
        filepath = self._filepath('artifacts', project, tag=tag)
        results = ArtifactList()
        results.tag = tag
        if isinstance(labels, str):
            labels = labels.split(',')
        if tag == '*':
            mask = '**/*' + name
            if name:
                mask += '*'
        else:
            mask = '**/*'
        for artifact, p in self._load_list(filepath, mask):
            if (name == '' or name in get_in(artifact, 'key', ''))\
                    and match_labels(get_in(artifact, 'labels', {}), labels):
                if 'artifacts/latest' in p:
                    artifact['tree'] = 'latest'
                results.append(artifact)

        return results

    def del_artifact(self, key, tag='', project=''):
        filepath = self._filepath('artifacts', project, key, tag) + self.format
        self._safe_del(filepath)

    def del_artifacts(self, name='', project='', tag='', labels=[]):
        tag = tag or 'latest'
        filepath = self._filepath('artifacts', project, tag=tag)

        if isinstance(labels, str):
            labels = labels.split(',')
        if tag == '*':
            mask = '**/*' + name
            if name:
                mask += '*'
        else:
            mask = '**/*'

        for artifact, p in self._load_list(filepath, mask):
            if (name == '' or name == get_in(artifact, 'key', ''))\
                    and match_labels(get_in(artifact, 'labels', {}), labels):

                self._safe_del(p)

    def _filepath(self, table, project, key='', tag=''):
        if tag == '*':
            tag = ''
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

    def _load_list(self, dirpath, mask):
        for p in pathlib.Path(dirpath).glob(mask + self.format):
            if p.is_file():
                if '.ipynb_checkpoints' in p.parts:
                    continue
                data = self._loads(p.read_text())
                if data:
                    yield data, str(p)

    def _safe_del(self, filepath):
        if path.isfile(filepath):
            remove(filepath)
        else:
            raise RunDBError(f'run file is not found or valid ({filepath})')
