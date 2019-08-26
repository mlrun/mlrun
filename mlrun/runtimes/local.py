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
import os
import inspect
from os import environ
from tempfile import mktemp

from ..model import RunObject
from ..execution import MLClientCtx
from .base import MLRuntime, RunError
from sys import executable, stderr
from subprocess import run, PIPE


class HandlerRuntime(MLRuntime):
    kind = 'handler'

    def _run(self, runobj: RunObject):
        self._force_handler()
        if self.rundb:
            environ['MLRUN_META_DBPATH'] = self.rundb

        args = inspect.signature(self.handler).parameters
        if len(args) > 1 and list(args.keys())[0] == 'context':
            # its a nuclio function
            from .function import fake_nuclio_context
            context, event = fake_nuclio_context(runobj.to_json())
            out = self.handler(context, event)
        elif len(args) >= 1:
            out = self.handler(runobj.to_dict())
        else:
            out = self.handler()

        if not out:
            return runobj
        if isinstance(out, MLClientCtx):
            return out.to_dict()
        if isinstance(out, dict):
            return out
        return json.loads(out)


class LocalRuntime(MLRuntime):
    kind = 'local'

    def _run(self, runobj: RunObject):
        environ['MLRUN_EXEC_CONFIG'] = runobj.to_json()
        tmp = mktemp('.json')
        environ['MLRUN_META_TMPFILE'] = tmp
        if self.rundb:
            environ['MLRUN_META_DBPATH'] = self.rundb

        cmd = [executable, self.runtime.command]
        args = self.runtime.args
        if args:
            cmd += args
        out = run(cmd, stdout=PIPE, stderr=PIPE)
        print(out.stdout.decode('utf-8'))
        if self.db_conn:
            uid = runobj.metadata.uid
            project = runobj.metadata.project or ''
            self.db_conn.store_log(uid, project, out.stdout.decode('utf-8'))
        if out.returncode != 0:
            print(out.stderr.decode('utf-8'), file=stderr)
            raise RunError(out.stderr.decode('utf-8'))

        try:
            with open(tmp) as fp:
                resp = fp.read()
            os.remove(tmp)
            if resp:
                return json.loads(resp)
        except FileNotFoundError as err:
            return runobj.to_dict()


