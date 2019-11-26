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
from os import path, remove, makedirs
import yaml
import pathlib
from datetime import datetime, timedelta

from ..utils import get_in, match_labels, dict_to_yaml, update_in, dict_to_json
from ..datastore import StoreManager
from .base import RunDBError, RunDBInterface
from ..lists import RunList, ArtifactList, FunctionList
from ..utils import logger

run_logs = 'runs'
artifacts_dir = 'artifacts'
functions_dir = 'functions'
default_project = 'default'
_missing = object()


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
        filepath = self._filepath(run_logs, project, uid, '') + '.log'
        makedirs(path.join(self.dirpath, run_logs, project), exist_ok=True)
        mode = 'ab' if append else 'wb'
        with open(filepath, mode) as fp:
            fp.write(body)
            fp.close()

    def get_log(self, uid, project='', offset=0, size=0):
        filepath = self._filepath(run_logs, project, uid, '') + '.log'
        if pathlib.Path(filepath).is_file():
            with open(filepath, 'rb') as fp:
                if offset:
                    fp.seek(offset)
                if not size:
                    size = 2**18
                return fp.read(size)
        return None

    def _run_path(self, uid, iter):
        if iter:
            return '{}-{}'.format(uid, iter)
        return uid

    def store_run(self, struct, uid, project='', iter=0):
        data = self._dumps(struct)
        filepath = self._filepath(
            run_logs, project, self._run_path(uid, iter), '') + self.format
        self._datastore.put(filepath, data)

    def update_run(self, updates: dict, uid, project='', iter=0):
        run = self.read_run(uid, project, iter=iter)
        if run and updates:
            for key, val in updates.items():
                update_in(run, key, val)
        self.store_run(run, uid, project, iter=iter)

    def read_run(self, uid, project='', iter=0):
        filepath = self._filepath(
            run_logs, project, self._run_path(uid, iter), '') + self.format
        data = self._datastore.get(filepath)
        return self._loads(data)

    def list_runs(self, name='', uid=None, project='', labels=None,
                  state='', sort=True, last=1000, iter=False):
        labels = [] if labels is None else labels
        filepath = self._filepath(run_logs, project)
        results = RunList()
        if isinstance(labels, str):
            labels = labels.split(',')
        for run, _ in self._load_list(filepath, '*'):
            if match_value(name, run, 'metadata.name') and \
               match_labels(get_in(run, 'metadata.labels', {}), labels) and \
               match_value(state, run, 'status.state') and \
               match_value(uid, run, 'metadata.uid'):
                results.append(run)

        if sort or last:
            results.sort(key=lambda i: get_in(
                i, ['status', 'start_time'], ''), reverse=True)
        if last and len(results) > last:
            return RunList(results[:last])
        return results

    def del_run(self, uid, project='', iter=0):
        filepath = self._filepath(
            run_logs, project, self._run_path(uid, iter), '') + self.format
        self._safe_del(filepath)

    def del_runs(self, name='', project='', labels=None, state='', days_ago=0):

        labels = [] if labels is None else labels
        if not name and not state and not days_ago:
            raise RunDBError(
                'filter is too wide, select name and/or state and/or days_ago')

        filepath = self._filepath(run_logs, project)
        if isinstance(labels, str):
            labels = labels.split(',')

        if days_ago:
            days_ago = datetime.now() - timedelta(days=days_ago)

        def date_before(run):
            return datetime.strptime(get_in(run, 'status.start_time', ''),
                                     '%Y-%m-%d %H:%M:%S.%f') < days_ago

        for run, p in self._load_list(filepath, '*'):
            if match_value(name, run, 'metadata.name') and \
               match_labels(get_in(run, 'metadata.labels', {}), labels) and \
               match_value(state, run, 'status.state') and \
               (not days_ago or date_before(run)):
                self._safe_del(p)

    def store_artifact(self, key, artifact, uid, tag='', project=''):
        artifact['updated'] = datetime.now()
        data = self._dumps(artifact)
        filepath = self._filepath(
            artifacts_dir, project, key, uid) + self.format
        self._datastore.put(filepath, data)
        filepath = self._filepath(
            artifacts_dir, project, key, tag or 'latest') + self.format
        self._datastore.put(filepath, data)

    def read_artifact(self, key, tag='', project=''):
        filepath = self._filepath(
            artifacts_dir, project, key, tag) + self.format
        data = self._datastore.get(filepath)
        return self._loads(data)

    def list_artifacts(self, name='', project='', tag='', labels=None):
        labels = [] if labels is None else labels
        tag = tag or 'latest'
        logger.info(
            f'reading artifacts in {project} name/mask: {name} tag: {tag} ...')
        filepath = self._filepath(artifacts_dir, project, tag=tag)
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
        filepath = self._filepath(
            artifacts_dir, project, key, tag) + self.format
        self._safe_del(filepath)

    def del_artifacts(self, name='', project='', tag='', labels=None):
        labels = [] if labels is None else labels
        tag = tag or 'latest'
        filepath = self._filepath(artifacts_dir, project, tag=tag)

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

    def store_function(self, func, name, project='', tag=''):
        update_in(func, 'metadata.updated', datetime.now())
        data = self._dumps(func)
        filepath = path.join(self.dirpath, '{}/{}/{}/{}'.format(
            functions_dir, project or default_project, name,
            tag or 'latest')) + self.format
        self._datastore.put(filepath, data)

    def get_function(self, name, project='', tag=''):
        filepath = path.join(self.dirpath, '{}/{}/{}/{}'.format(
            functions_dir, project or default_project, name,
            tag or 'latest')) + self.format
        data = self._datastore.get(filepath)
        return self._loads(data)

    def list_functions(self, name, project='', tag='', labels=None):
        labels = labels or []
        logger.info(
            f'reading functions in {project} name/mask: {name} tag: {tag} ...')
        filepath = path.join(self.dirpath, '{}/{}/'.format(
            functions_dir, project or default_project))
        results = []
        if isinstance(labels, str):
            labels = labels.split(',')
        mask = '**/*'
        if name:
            filepath = '{}{}/'.format(filepath, name)
            mask = '*'
        for func, p in self._load_list(filepath, mask):
            if match_labels(get_in(func, 'metadata.labels', {}), labels):
                results.append(func)

        return results

    def _filepath(self, table, project, key='', tag=''):
        if tag == '*':
            tag = ''
        if tag:
            key = '/' + key
        project = project or default_project
        return path.join(self.dirpath, '{}/{}/{}{}'.format(
            table, project, tag, key))

    _encodings = {
        '.yaml': ('to_yaml', dict_to_yaml),
        '.json': ('to_json', dict_to_json),
    }

    def _dumps(self, obj):
        meth_name, enc_fn = self._encodings.get(self.format, (None, None))
        if meth_name is None:
            raise ValueError(f'unsupported format - {self.format}')

        meth = getattr(obj, meth_name, None)
        if meth:
            return meth()

        return enc_fn(obj)

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


def match_value(value, obj, key):
    if not value:
        return True
    return get_in(obj, key, _missing) == value
