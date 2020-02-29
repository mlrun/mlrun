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

from ..config import config
from .base import RunDBError, RunDBInterface  # noqa
from .filedb import FileRunDB
from .httpdb import HTTPRunDB
from .sqldb import SQLDB
from os import environ


def get_or_set_dburl(default=''):
    if not config.dbpath and default:
        config.dbpath = default
        environ['MLRUN_DBPATH'] = default
    return config.dbpath


def get_run_db(url=''):
    if not url:
        url = get_or_set_dburl('./')

    p = urlparse(url)
    scheme = p.scheme.lower()
    if '://' not in url or scheme in ['file', 's3', 'v3io', 'v3ios']:
        cls = FileRunDB
    elif scheme in ('http', 'https'):
        cls = HTTPRunDB
    else:
        cls = SQLDB

    return cls(url)
