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

from base64 import b64encode
from os import path, environ, makedirs
from shutil import copyfile
from urllib.parse import urlparse
from .utils import run_keys
import boto3
import requests

V3IO_LOCAL_ROOT = 'v3io'


def get_object(url, secrets=None):
    stores = StoreManager(secrets)
    datastore, subpath = stores.get_or_create_store(url)
    return datastore.get(subpath)


def parseurl(url):
    p = urlparse(url)
    schema = p.scheme.lower()
    endpoint = p.hostname
    if p.port:
        endpoint += ':{}'.format(p.port)
    return schema, endpoint, p.path


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
    schema, endpoint, subpath = parseurl(link)
    return schema_to_store(schema).uri_to_ipython(endpoint, subpath)


class StoreManager:
    def __init__(self, secrets=None):
        self._stores = {}
        self._secrets = secrets or {}

    def from_dict(self, struct: dict):
        stor_list = struct.get(run_keys.data_stores)
        if stor_list and isinstance(stor_list, list):
            for stor in stor_list:
                schema, endpoint, subpath = parseurl(stor.get('url'))
                new_stor = schema_to_store(schema)(self, schema, stor['name'], endpoint)
                new_stor.subpath = subpath
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

    def object(self, key, url=''):
        store, subpath = self.get_or_create_store(url)
        return DataItem(key, store, subpath, url)

    def get_or_create_store(self, url):
        schema, endpoint, subpath = parseurl(url)

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


class DataStore:
    def __init__(self, parent: StoreManager, name, kind, endpoint=''):
        self._parent = parent
        self.kind = kind
        self.name = name
        self.endpoint = endpoint
        self.subpath = ''
        self.secret_pfx = ''
        self.options = {}
        self.from_spec = False

    @property
    def is_structured(self):
        return False

    @property
    def is_unstructured(self):
        return True

    @staticmethod
    def uri_to_kfp(endpoint, subpath):
        raise ValueError('data store doesnt support KFP URLs')

    @staticmethod
    def uri_to_ipython(endpoint, subpath):
        return ''

    def _join(self, key):
        if self.subpath:
            return '{}/{}'.format(self.subpath, key)
        return key

    def _secret(self, key):
        return self._parent.secret(self.secret_pfx + key)

    @property
    def url(self):
        return '{}://{}'.format(self.kind, self.endpoint)

    def get(self, key):
        pass

    def query(self, key, query='', **kwargs):
        raise ValueError('data store doesnt support structured queries')

    def put(self, key, data, append=False):
        pass

    def download(self, key, target_path):
        data = self.get(key)
        mode = 'wb'
        if isinstance(data, str):
            mode = 'w'
        with open(target_path, mode) as fp:
            fp.write(data)
            fp.close()

    def upload(self, key, src_path):
        pass

    def to_dict(self):
        return {
                'name': self.name,
                'url': '{}://{}/{}'.format(self.kind, self.endpoint, self.subpath),
                'secret_pfx': self.secret_pfx,
                'options': self.options,
            }


class DataItem:
    def __init__(self, key, store, subpath, url=''):
        self._store = store
        self._key = key
        self._url = url
        self._path = subpath

    @property
    def kind(self):
        return self._store.kind

    @property
    def url(self):
        return self._url

    def get(self):
        return self._store.get(self._path)

    def download(self, target_path):
        self._store.download(self._path, target_path)

    def put(self, data):
        self._store.put(self._path, data)

    def upload(self, src_path):
        self._store.upload(self._path, src_path)

    def __str__(self):
        return self.url

    def __repr__(self):
        return "'{}'".format(self.url)


class FileStore(DataStore):
    def __init__(self, parent: StoreManager, schema, name, endpoint=''):
        super().__init__(parent, name, 'file', endpoint)

    @property
    def url(self):
        return self.subpath

    def _join(self, key):
        return path.join(self.subpath, key)

    def get(self, key):
        with open(self._join(key), 'rb') as fp:
            return fp.read()

    def put(self, key, data, append=False):
        dir = path.dirname(self._join(key))
        if dir:
            makedirs(dir, exist_ok=True)
        mode = 'a' if append else 'w'
        if isinstance(data, bytes):
            mode = mode + 'b'
        with open(self._join(key), mode) as fp:
            fp.write(data)
            fp.close()

    def download(self, key, target_path):
        fullpath = self._join(key)
        if fullpath == target_path:
            return
        copyfile(fullpath, target_path)

    def upload(self, key, src_path):
        fullpath = self._join(key)
        if fullpath == src_path:
            return
        dir = path.dirname(fullpath)
        if dir:
            makedirs(dir, exist_ok=True)
        copyfile(src_path, fullpath)


class S3Store(DataStore):
    def __init__(self, parent: StoreManager, schema, name, endpoint=''):
        super().__init__(parent, name, schema, endpoint)
        region = None

        access_key = self._secret('AWS_ACCESS_KEY_ID')
        secret_key = self._secret('AWS_SECRET_ACCESS_KEY')

        if access_key or secret_key:
            self.s3 = boto3.resource('s3', region_name=region,
                                     aws_access_key_id=access_key,
                                     aws_secret_access_key=secret_key)
        else:
            # from env variables
            self.s3 = boto3.resource('s3', region_name=region)

    def upload(self, key, src_path):
        self.s3.Object(self.endpoint, self._join(key)[1:]).put(Body=open(src_path, 'rb'))

    def get(self, key):
        obj = self.s3.Object(self.endpoint, self._join(key)[1:])
        return obj.get()['Body'].read()

    def put(self, key, data, append=False):
        self.s3.Object(self.endpoint, self._join(key)[1:]).put(Body=data)


def basic_auth_header(user, password):
    username = user.encode('latin1')
    password = password.encode('latin1')
    base = b64encode(b':'.join((username, password))).strip()
    authstr = 'Basic ' + base.decode('ascii')
    return {'Authorization': authstr}


def http_get(url, headers=None, auth=None):
    try:
        resp = requests.get(url, headers=headers, auth=auth, verify=False)
    except OSError:
        raise OSError('error: cannot connect to {}'.format(url))

    if not resp.ok:
        raise OSError('failed to read file in {}'.format(url))
    return resp.content


def http_put(url, data, headers=None, auth=None):
    try:
        resp = requests.put(url, data=data, headers=headers, auth=auth, verify=False)
    except OSError:
        raise OSError('error: cannot connect to {}'.format(url))
    if not resp.ok:
        raise OSError(
            'failed to upload to {} {}'.format(url, resp.status_code))


def http_upload(url, file_path, headers=None, auth=None):
    with open(file_path, 'rb') as data:
        http_put(url, data, headers, auth)


class HttpStore(DataStore):
    def __init__(self, parent: StoreManager, schema, name, endpoint=''):
        super().__init__(parent, name, schema, endpoint)
        self.auth = None

    def upload(self, key, src_path):
        raise ValueError('unimplemented')

    def put(self, key, data, append=False):
        raise ValueError('unimplemented')

    def get(self, key):
        return http_get(self.url + self._join(key), None, self.auth)


class V3ioStore(DataStore):
    def __init__(self, parent: StoreManager, schema, name, endpoint=''):
        super().__init__(parent, name, schema, endpoint)
        self.endpoint = self.endpoint or environ.get('V3IO_API')

        token = self._secret('V3IO_ACCESS_KEY') or environ.get('V3IO_ACCESS_KEY')
        username = self._secret('V3IO_USERNAME') or environ.get('V3IO_USERNAME')
        password = self._secret('V3IO_PASSWORD') or environ.get('V3IO_PASSWORD')

        self.headers = None
        self.auth = None
        if token:
            self.headers = {'X-v3io-session-key': token}
        elif username and password:
            self.headers = basic_auth_header(username, password)

    @staticmethod
    def uri_to_ipython(endpoint, subpath):
        return V3IO_LOCAL_ROOT + subpath

    @property
    def url(self):
        schema = 'http' if self.kind == 'v3io' else 'https'
        return '{}://{}'.format(schema, self.endpoint)

    def upload(self, key, src_path):
        http_upload(self.url + self._join(key), src_path, self.headers, None)

    def get(self, key):
        return http_get(self.url + self._join(key), self.headers, None)

    def put(self, key, data, append=False):
        http_put(self.url + self._join(key), data, self.headers, None)
