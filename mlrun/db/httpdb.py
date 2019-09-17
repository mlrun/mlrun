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


import requests

from .base import RunDBError, RunDBInterface

default_project = 'default'  # TODO: Name?

_artifact_keys = [
    'format',
    'inline',
    'key',
    'src_path',
    'target_path',
    'viewer',
]


def bool2str(val):
    return 'yes' if val else 'no'


class HTTPRunDB(RunDBInterface):
    def __init__(self, base_url):
        self.base_url = base_url

    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls}({self.base_url!r})'

    def _api_call(self, method, path, error=None, params=None, body=None):
        url = f'{self.base_url}/{path}'
        kw = {
            key: value
            for key, value in (('params', params), ('data', body))
            if value is not None
        }

        try:
            resp = requests.request(method, url, **kw)
            resp.raise_for_status()
            return resp
        except requests.RequestException as err:
            error = error or '{method} {url}'
            raise RunDBError(error) from err

    def _path_of(self, prefix, project, uid):
        project = project or default_project
        return f'{prefix}/{project}/{uid}'

    def connect(self, secrets=None):
        self._api_call('GET', 'healthz')

    def store_log(self, uid, project='', body=None, append=True):
        if not body:
            return

        path = self._path_of('log', project, uid)
        params = {'append': bool2str(append)}
        error = f'store log {project}/{uid}'
        self._api_call('POST', path, error, params, body)

    def get_log(self, uid, project=''):
        path = self._path_of('log', project, uid)
        error = f'get log {project}/{uid}'
        resp = self._api_call('GET', path, error)
        return resp.content

    def store_run(self, struct, uid, project='', commit=False):
        path = self._path_of('run', project, uid)
        error = f'store run {project}/{uid}'
        params = {'commit': bool2str(commit)}
        body = json.dumps(struct)
        self._api_call('POST', path, error, params, body=body)

    def update_run(self, updates: dict, uid, project=''):
        path = self._path_of('run', project, uid)
        error = f'update run {project}/{uid}'
        body = json.dumps(updates)
        self._api_call('PATCH', path, error, body=body)

    def read_run(self, uid, project=''):
        path = self._path_of('run', project, uid)
        error = f'get run {project}/{uid}'
        resp = self._api_call('GET', path, error)
        return resp.json()['data']

    def del_run(self, uid, project=''):
        path = self._path_of('run', project, uid)
        error = f'del run {project}/{uid}'
        self._api_call('DELETE', path, error)

    def list_runs(
            self, name='', project='', labels=None, state='', sort=True,
            last=0):

        params = {
            'name': name,
            'project': project,
            'label': labels or [],
            'state': state,
            'sort': bool2str(sort),
        }
        error = 'list runs'
        resp = self._api_call('GET', 'runs', error, params=params)
        return resp.json()['runs']

    def del_runs(self, name='', project='', labels=None, state='', days_ago=0):
        params = {
            'name': name,
            'project': project,
            'label': labels or [],
            'state': state,
            'days_ago': str(days_ago),
        }
        error = 'del runs'
        self._api_call('DELETE', 'runs', error, params=params)

    def store_artifact(self, key, artifact, uid, tag='', project=''):
        path = self._path_of('artifact', project, uid)
        params = {
            'key': key,
            'tag': tag,
        }

        data = {key: getattr(artifact, key) for key in _artifact_keys}
        data['body'] = artifact.get_body()
        error = f'store artifact {project}/{uid}'
        self._api_call(
            'POST', path, error, params=params, body=json.dumps(data))

    def read_artifact(self, key, tag='', project=''):
        path = self._path_of('artifact', project, key)  # TODO: uid?
        params = {
            'key': key,
            'tag': tag,
        }
        error = f'read artifact {project}/{key}'
        resp = self._api_call('GET', path, error, params=params)
        return resp.content

    def del_artifact(self, key, tag='', project=''):
        path = self._path_of('artifact', project, key)  # TODO: uid?
        params = {
            'key': key,
            'tag': tag,
        }
        error = f'del artifact {project}/{key}'
        self._api_call('DELETE', path, error, params=params)

    def list_artifacts(self, name='', project='', tag='', labels=None):
        params = {
            'name': name,
            'project': project,
            'tag': tag,
            'label': labels or [],
        }
        error = 'list artifacts'
        resp = self._api_call('GET', 'artifacts', error, params=params)
        return resp.json()['artifacts']

    def del_artifacts(
            self, name='', project='', tag='', labels=None, days_ago=0):
        params = {
            'name': name,
            'project': project,
            'tag': tag,
            'label': labels or [],
            'days_ago': str(days_ago),
        }
        error = 'del artifacts'
        self._api_call('DELETE', 'artifacts', error, params=params)

    def store_metric(
            self, uid, project='', keyvals=None, timestamp=None, labels=None):
        raise NotImplementedError('store_metric')

    def read_metric(self, keys, project='', query=''):
        raise NotImplementedError('read_metric')
