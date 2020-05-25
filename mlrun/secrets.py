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

from ast import literal_eval
from os import environ
import pickle

from cryptography.fernet import Fernet

from .utils import list2dict

_env_enc_key = 'MLRUN_SECRETS_KEY'


class SecretsStore:
    def __init__(self):
        self._secrets = {}

    @classmethod
    def from_list(cls, src_list: list):
        store = cls()
        if src_list and isinstance(src_list, list):
            for src in src_list:
                store.add_source(
                    src['kind'], src.get('source'), src.get('prefix', ''))
        return store

    def to_dict(self, struct):
        pass

    def add_source(self, kind, source='', prefix=''):

        if kind == 'inline':
            if isinstance(source, str):
                source = literal_eval(source)
            if not isinstance(source, dict):
                raise ValueError("inline secrets must be of type dict")
            for k, v in source.items():
                self._secrets[prefix + k] = str(v)

        elif kind == 'file':
            with open(source) as fp:
                lines = fp.read().splitlines()
                secrets_dict = list2dict(lines)
                for k, v in secrets_dict.items():
                    self._secrets[prefix + k] = str(v)

        elif kind == 'env':
            for key in source.split(','):
                k = key.strip()
                self._secrets[prefix + k] = environ.get(k)

    def get(self, key, default=None):
        return self._secrets.get(key, default)

    def get_all(self):
        return self._secrets.copy()

    def to_serial(self):
        # todo: use encryption
        return [{'kind': 'inline', 'source': self._secrets.copy()}]


def new_key():
    """Return a new encryption key"""
    return Fernet.generate_key()


def _enc(value, key):
    enc = Fernet(key)
    data = enc.encrypt(pickle.dumps(value))
    return data.decode('utf-8')


def encrypt_dict(d: dict, key=None):
    """Encrypt values in a dictionary"""
    enc_key = _resolve_key(key)
    return {key: _enc(val, enc_key) for key, val in d.items()}


def _dec(value, key):
    if isinstance(value, str):
        value = value.encode('utf-8')
    dec = Fernet(key)
    return pickle.loads(dec.decrypt(value))


def decrypt_dict(d: dict, key=None):
    """Decrypt values in a dictionary"""
    enc_key = _resolve_key(key)
    return {key: _dec(val, enc_key) for key, val in d.items()}


def _resolve_key(key):
    if key:
        return key

    key = environ.get(_env_enc_key)
    if not key:
        raise ValueError(f'key is None and {_env_enc_key} not set')
    return key.encode('utf-8')
