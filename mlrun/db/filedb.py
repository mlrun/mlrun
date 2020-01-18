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
import pathlib
from datetime import datetime, timedelta
from os import makedirs, path, remove, scandir

import yaml

from ..config import config
from ..datastore import StoreManager
from ..lists import ArtifactList, RunList
from ..utils import (
    dict_to_json, dict_to_yaml, get_in, logger, match_labels, match_value,
    update_in
)
from .base import RunDBError, RunDBInterface

run_logs = 'runs'
artifacts_dir = 'artifacts'
functions_dir = 'functions'
schedules_dir = 'schedules'


class FileRunDB(RunDBInterface):
    kind = 'file'

    def __init__(self, dirpath='', format='.yaml'):
        self.format = format
        self.dirpath = dirpath
        self._datastore = None
        self._subpath = None
        makedirs(self.schedules_dir, exist_ok=True)

    def connect(self, secrets=None):
        sm = StoreManager(secrets)
        self._datastore, self._subpath = sm.get_or_create_store(self.dirpath)
        return self

    def store_log(self, uid, project='', body=None, append=False):
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
                return '', fp.read(size)
        return '', None

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
        # TODO: Should we raise if run not found?
        if run and updates:
            for key, val in updates.items():
                update_in(run, key, val)
        self.store_run(run, uid, project, iter=iter)

    def read_run(self, uid, project='', iter=0):
        filepath = self._filepath(
            run_logs, project, self._run_path(uid, iter), '') + self.format
        if not pathlib.Path(filepath).is_file():
            return None
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
        if not pathlib.Path(filepath).is_file():
            return None
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
            functions_dir, project or config.default_project, name,
            tag or 'latest')) + self.format
        self._datastore.put(filepath, data)

    def get_function(self, name, project='', tag=''):
        filepath = path.join(self.dirpath, '{}/{}/{}/{}'.format(
            functions_dir, project or config.default_project, name,
            tag or 'latest')) + self.format
        if not pathlib.Path(filepath).is_file():
            return None
        data = self._datastore.get(filepath)
        return self._loads(data)

    def list_functions(self, name, project='', tag='', labels=None):
        labels = labels or []
        logger.info(
            f'reading functions in {project} name/mask: {name} tag: {tag} ...')
        filepath = path.join(self.dirpath, '{}/{}/'.format(
            functions_dir, project or config.default_project))
        results = []
        if isinstance(labels, str):
            labels = labels.split(',')
        mask = '**/*'
        if name:
            filepath = '{}{}/'.format(filepath, name)
            mask = '*'
        for func, _ in self._load_list(filepath, mask):
            if match_labels(get_in(func, 'metadata.labels', {}), labels):
                results.append(func)

        return results

    def _filepath(self, table, project, key='', tag=''):
        if tag == '*':
            tag = ''
        if tag:
            key = '/' + key
        project = project or config.default_project
        return path.join(self.dirpath, '{}/{}/{}{}'.format(
            table, project, tag, key))

    @property
    def schedules_dir(self):
        return path.join(self.dirpath, schedules_dir)

    def store_schedule(self, data):
        sched_id = 1 + sum(1 for _ in scandir(self.schedules_dir))
        fname = path.join(
            self.schedules_dir,
            '{}{}'.format(sched_id, self.format),
        )
        with open(fname, 'w') as out:
            out.write(self._dumps(data))

    def list_schedules(self):
        pattern = '*{}'.format(self.format)
        for p in pathlib.Path(self.schedules_dir).glob(pattern):
            with p.open() as fp:
                yield self._loads(fp.read())

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
