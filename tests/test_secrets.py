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

from contextlib import contextmanager
from os import environ

import pytest

from mlrun import secrets

spec = {
    'secret_sources': [
        {'kind': 'file', 'source': 'tests/secrets_test.txt'},
        {'kind': 'inline', 'source': {'abc': 'def'}},
        {'kind': 'env', 'source': 'ENV123,ENV456'},
    ],
}


def test_load():
    environ['ENV123'] = 'xx'
    environ['ENV456'] = 'yy'
    ss = secrets.SecretsStore.from_list(spec['secret_sources'])

    assert ss.get('ENV123') == 'xx', 'failed on 1st env var secret'
    assert ss.get('ENV456') == 'yy', 'failed on 1st env var secret'
    assert ss.get('MYENV') == '123', 'failed on 1st env var secret'
    assert ss.get('MY2NDENV') == '456', 'failed on 1st env var secret'
    assert ss.get('abc') == 'def', 'failed on 1st env var secret'
    print(ss.get_all())


def test_inline_str():
    spec = {
        'secret_sources': [
            {'kind': 'inline', 'source': "{'abc': 'def'}"},
        ],
    }

    ss = secrets.SecretsStore.from_list(spec['secret_sources'])
    assert ss.get('abc') == 'def', 'failed on 1st env var secret'


@contextmanager
def tmp_env(kw=None, rm=None):
    kw = {} if kw is None else kw
    rm = [] if rm is None else rm
    orig = {key: environ.get(key) for key in kw if key in environ}
    removed = {}
    for key in rm:
        if key in environ:
            removed[key] = environ.pop(key)
    environ.update(kw)

    yield

    for key in kw:
        val = orig.get(key)
        if val:
            environ[key] = val
        else:
            del environ[key]
    for key, val in removed.items():
        environ[key] = val


def test_encrypt():
    d = {'a': 1, 'b': 1.3, 'c': {'a', None}, 'd': None}
    key = secrets.new_key()

    enc = secrets.encrypt_dict(d, key)
    assert set(enc) == set(d), 'keys mismatch'

    dec = secrets.decrypt_dict(enc, key)
    assert d == dec, 'bad decrypt'

    env = {secrets._env_enc_key: key.decode('utf-8')}
    with tmp_env(env):
        enc2 = secrets.encrypt_dict(d)
        dec2 = secrets.decrypt_dict(enc2)
        assert dec2 == d, 'bad encrypt from env'

    with tmp_env(rm=[secrets._env_enc_key]), pytest.raises(ValueError):
        secrets.encrypt_dict(d)
