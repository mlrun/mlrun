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

from .base import RunError
from sys import stderr
from ..utils import logger


def log_std(db, runobj, out, err='', skip=False):
    line = '> ' + '-' * 15 + ' Iteration: ({}) ' + '-' * 15 + '\n'
    if out:
        iter = runobj.metadata.iteration
        if iter:
            out = line.format(iter) + out
        print(out)
        if db and not skip:
            uid = runobj.metadata.uid
            project = runobj.metadata.project or ''
            db.store_log(uid, project, out.encode(), append=True)
    if err:
        logger.error('exec error - {}'.format(err))
        print(err, file=stderr)
        raise RunError(err)


class AsyncLogWriter:
    def __init__(self, db, runobj):
        self.db = db
        self.uid = runobj.metadata.uid
        self.project = runobj.metadata.project or ''
        self.iter = runobj.metadata.iteration

    def write(self, data):
        if self.db:
            self.db.store_log(self.uid, self.project, data, append=True)

    def flush(self):
        # todo: verify writes are large enough, if not cache and use flush
        pass