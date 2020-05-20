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
from os import remove, path
from tempfile import mktemp

import requests
import urllib3
import pandas as pd

from mlrun.utils import logger

verify_ssl = False
if not verify_ssl:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class FileStats:
    def __init__(self, size, modified, content_type=None):
        self.size = size
        self.modified = modified
        self.content_type = content_type


class DataStore:
    def __init__(self, parent, name, kind, endpoint=''):
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

    def get(self, key, size=None, offset=0):
        pass

    def query(self, key, query='', **kwargs):
        raise ValueError('data store doesnt support structured queries')

    def put(self, key, data, append=False):
        pass

    def stat(self, key):
        pass

    def listdir(self, key):
        raise ValueError('data store doesnt support listdir')

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

    def as_df(self, key, columns=None, df_module=None, format='', **kwargs):
        df_module = df_module or pd
        if key.endswith(".csv") or format == 'csv':
            if columns:
                kwargs['usecols'] = columns
            reader = df_module.read_csv
        elif key.endswith(".parquet") or key.endswith(".pq") or format == 'parquet':
            if columns:
                kwargs['columns'] = columns
            reader = df_module.read_parquet
        elif key.endswith(".json") or format == 'json':
            reader = df_module.read_json

        else:
            raise Exception(f"file type unhandled {key}")

        if self.kind == 'file':
            return reader(self._join(key), **kwargs)

        tmp = mktemp()
        self.download(self._join(key), tmp)
        df = reader(tmp, **kwargs)
        remove(tmp)
        return df

    def to_dict(self):
        return {
                'name': self.name,
                'url': '{}://{}/{}'.format(self.kind, self.endpoint, self.subpath),
                'secret_pfx': self.secret_pfx,
                'options': self.options,
            }


class DataItem:
    def __init__(self, key: str, store: DataStore, subpath: str,
                 url: str = '', meta=None, artifact_url=None):
        self._store = store
        self._key = key
        self._url = url
        self._path = subpath
        self._meta = meta
        self._artifact_url = artifact_url
        self._local_path = ''

    @property
    def key(self):
        return self._key

    @property
    def suffix(self):
        _, file_ext = path.splitext(self._path)
        return file_ext

    @property
    def kind(self):
        return self._store.kind

    @property
    def meta(self):
        return self._meta

    @property
    def artifact_url(self):
        return self._artifact_url or self._url

    @property
    def url(self):
        return self._url

    def get(self, size=None, offset=0):
        return self._store.get(self._path, size=size, offset=offset)

    def download(self, target_path):
        self._store.download(self._path, target_path)

    def put(self, data, append=False):
        self._store.put(self._path, data, append=append)

    def upload(self, src_path):
        self._store.upload(self._path, src_path)

    def stat(self):
        return self._store.stat(self._path)

    def listdir(self):
        return self._store.listdir(self._path)

    def local(self):
        if self.kind == 'file':
            return self._path
        if self._local_path:
            return self._local_path

        dot = self._path.rfind('.')
        self._local_path = mktemp() if dot == -1 else \
            mktemp(self._path[dot:])
        logger.info('downloading {} to local tmp'.format(self.url))
        self.download(self._local_path)
        return self._local_path

    def as_df(self, columns=None, df_module=None, format='', **kwargs):
        return self._store.as_df(self._path, columns=columns,
                                 df_module=df_module, format=format, **kwargs)

    def __str__(self):
        return self.url

    def __repr__(self):
        return "'{}'".format(self.url)


def get_range(size, offset):
    byterange = 'bytes={}-'.format(offset)
    if size:
        byterange = range + '{}'.format(offset + size)
    return byterange


def basic_auth_header(user, password):
    username = user.encode('latin1')
    password = password.encode('latin1')
    base = b64encode(b':'.join((username, password))).strip()
    authstr = 'Basic ' + base.decode('ascii')
    return {'Authorization': authstr}


def http_get(url, headers=None, auth=None):
    try:
        resp = requests.get(url, headers=headers, auth=auth, verify=verify_ssl)
    except OSError as e:
        raise OSError('error: cannot connect to {}: {}'.format(url, e))

    if not resp.ok:
        raise OSError('failed to read file in {}'.format(url))
    return resp.content


def http_head(url, headers=None, auth=None):
    try:
        resp = requests.head(url, headers=headers, auth=auth, verify=verify_ssl)
    except OSError as e:
        raise OSError('error: cannot connect to {}: {}'.format(url, e))
    if not resp.ok:
        raise OSError('failed to read file head in {}'.format(url))
    return resp.headers


def http_put(url, data, headers=None, auth=None):
    try:
        resp = requests.put(url, data=data, headers=headers,
                            auth=auth, verify=verify_ssl)
    except OSError as e:
        raise OSError('error: cannot connect to {}: {}'.format(url, e))
    if not resp.ok:
        raise OSError(
            'failed to upload to {} {}'.format(url, resp.status_code))


def http_upload(url, file_path, headers=None, auth=None):
    with open(file_path, 'rb') as data:
        http_put(url, data, headers, auth)


class HttpStore(DataStore):
    def __init__(self, parent, schema, name, endpoint=''):
        super().__init__(parent, name, schema, endpoint)
        self.auth = None

    def upload(self, key, src_path):
        raise ValueError('unimplemented')

    def put(self, key, data, append=False):
        raise ValueError('unimplemented')

    def get(self, key, size=None, offset=0):
        data = http_get(self.url + self._join(key), None, self.auth)
        if offset:
            data = data[offset:]
        if size:
            data = data[:size]
        return data