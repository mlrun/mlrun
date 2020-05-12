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

from urllib.parse import urlparse
import mlrun

from ..config import config
from ..utils import run_keys, DB_SCHEMA

from .base import DataItem, HttpStore
from .s3 import S3Store
from .filestore import FileStore
from .v3io import V3ioStore


def get_object_stat(url, secrets=None):
    stores = StoreManager(secrets)
    return stores.object(url=url).stat()


def parseurl(url):
    p = urlparse(url)
    schema = p.scheme.lower()
    endpoint = p.hostname
    if p.port:
        endpoint += ':{}'.format(p.port)
    return schema, endpoint, p


def schema_to_store(schema):
    if not schema or schema in ['file', 'c', 'd']:
        return FileStore
    elif schema == 's3':
        return S3Store
    elif schema in ['v3io', 'v3ios']:
        return V3ioStore
    elif schema in ['http', 'https']:
        return HttpStore
    else:
        raise ValueError('unsupported store scheme ({})'.format(schema))


def uri_to_ipython(link):
    schema, endpoint, p = parseurl(link)
    if schema == DB_SCHEMA:
        return ''
    return schema_to_store(schema).uri_to_ipython(endpoint, p.path)


class StoreManager:
    def __init__(self, secrets=None, db=None):
        self._stores = {}
        self._secrets = secrets or {}
        self._db = db

    def _get_db(self):
        if not self._db:
            self._db = mlrun.get_run_db().connect(self._secrets)
        return self._db

    def from_dict(self, struct: dict):
        stor_list = struct.get(run_keys.data_stores)
        if stor_list and isinstance(stor_list, list):
            for stor in stor_list:
                schema, endpoint, p = parseurl(stor.get('url'))
                new_stor = schema_to_store(schema)(self, schema, stor['name'], endpoint)
                new_stor.subpath = p.path
                new_stor.secret_pfx = stor.get('secret_pfx')
                new_stor.options = stor.get('options', {})
                new_stor.from_spec = True
                self._stores[stor['name']] = new_stor

    def to_dict(self, struct):
        struct[run_keys.data_stores] = [stor.to_dict() for stor in self._stores.values() if stor.from_spec]

    def secret(self, key):
        return self._secrets.get(key)

    def _add_store(self, store):
        self._stores[store.name] = store

    def get_store_artifact(self, url, project=''):
        schema, endpoint, p = parseurl(url)
        if not p.path:
            raise ValueError('store url without a path {}'.format(url))
        key = p.path[1:]
        project = endpoint or project or config.default_project
        tag = p.fragment if p.fragment else ''
        iteration = None
        if '/' in key:
            idx = key.rfind('/')
            try:
                iteration = int(key[idx+1:])
            except ValueError:
                raise ValueError('illegal store path {}, iteration must be integer value'.format(url))
            key = key[:idx]

        try:
            meta = self._get_db().read_artifact(key, tag=tag,
                                                iter=iteration,
                                                project=project)
            if meta.get('kind', '') == 'link':
                # todo: support other link types (not just iter, move this to the db/api layer
                meta = self._get_db().read_artifact(p.path[1:],
                                                    tag=tag,
                                                    iter=meta.get('link_iteration', 0),
                                                    project=project)

            meta = mlrun.artifacts.dict_to_artifact(meta)
        except Exception as e:
            raise OSError('artifact {} not found, {}'.format(url, e))
        return meta, meta.target_path

    def object(self, url, key='', project=''):
        meta = artifact_url = None
        if url.startswith(DB_SCHEMA + '://'):
            artifact_url = url
            meta, url = self.get_store_artifact(url, project)

        store, subpath = self.get_or_create_store(url)
        return DataItem(key, store, subpath, url,
                        meta=meta, artifact_url=artifact_url)

    def get_or_create_store(self, url):
        schema, endpoint, p = parseurl(url)
        subpath = p.path

        if not schema and endpoint:
            if endpoint in self._stores.keys():
                return self._stores[endpoint], subpath
            else:
                raise ValueError('no such store ({})'.format(endpoint))

        storekey = '{}://{}'.format(schema, endpoint)
        if storekey in self._stores.keys():
            return self._stores[storekey], subpath

        store = schema_to_store(schema)(self, schema, storekey, endpoint)
        self._stores[storekey] = store
        return store, subpath


