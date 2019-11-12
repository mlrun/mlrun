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
from pathlib import Path
from sys import exc_info

sql_dir = Path(__file__).absolute().parent / 'sql'


def load_sql(name):
    path = sql_dir / name
    with path.open() as fp:
        return fp.read()


@contextmanager
def transaction(conn):
    cur = conn.cursor()
    try:
        yield cur
    finally:
        cur.close()
        _, err, _ = exc_info()
        if err:
            conn.rollback()
        else:
            conn.commit()


def initialize(conn):
    sql = load_sql('schema.sql')
    with transaction(conn) as cur:
        cur.execute(sql)
