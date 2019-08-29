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

import importlib.util as imputil
from io import StringIO
from contextlib import redirect_stdout
from pathlib import Path


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
        if self.runtime.handler:
            val, sout, serr = run_func(self.runtime.command,
                                       self.runtime.handler)
        else:
            val, sout, serr = run_exec(self.runtime.command,
                                       self.runtime.args)

        if self.db_conn:
            uid = runobj.metadata.uid
            project = runobj.metadata.project or ''
            self.db_conn.store_log(uid, project, sout)
        if serr:
            print(serr, file=stderr)
            raise RunError(serr)

        try:
            with open(tmp) as fp:
                resp = fp.read()
            os.remove(tmp)
            if resp:
                return json.loads(resp)
        except FileNotFoundError as err:
            return runobj.to_dict()


def load_module(file_name):
    """Load module from file name"""
    path = Path(file_name)
    mod_name = path.name
    if path.suffix:
        mod_name = mod_name[:-len(path.suffix)]
    spec = imputil.spec_from_file_location(mod_name, file_name)
    if spec is None:
        raise ImportError(f'cannot import from {file_name!r}')
    mod = imputil.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_exec(command, args):
    cmd = [executable, command]
    if args:
        cmd += args
    out = run(cmd, stdout=PIPE, stderr=PIPE)
    print(out.stdout.decode('utf-8'))

    err = out.stderr.decode('utf-8') if out.returncode != 0 else ''
    return None, out.stdout.decode('utf-8'), err


def run_func(file_name, name='main', args=None, kw=None, *, ctx=None):
    """Run a function from file with args and kw.

    ctx values are injected to module during function run time.
    """
    mod = load_module(file_name)
    fn = getattr(mod, name)  # Will raise if name not found

    if ctx is not None:
        for attr, value in ctx.items():
            setattr(mod, attr, value)

    args = [] if args is None else args
    kw = {} if kw is None else kw

    stdout = StringIO()
    err = ''
    val = None
    with redirect_stdout(stdout):
        try:
            val = fn(*args, **kw)
        except Exception as e:
            err = str(e)

    return val, stdout.getvalue(), err
