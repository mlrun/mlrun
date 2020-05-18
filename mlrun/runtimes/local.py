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
import inspect
import os
import socket
import sys
import traceback
from copy import copy
from os import environ, remove
from tempfile import mktemp

from .kubejob import KubejobRuntime
from ..model import RunObject
from ..utils import logger
from ..execution import MLClientCtx
from .base import BaseRuntime
from .utils import log_std, global_context, RunError
from sys import executable
from subprocess import run, PIPE

import importlib.util as imputil
from io import StringIO
from contextlib import redirect_stdout
from pathlib import Path
from nuclio import Event


class HandlerRuntime(BaseRuntime):
    kind = 'handler'

    def _run(self, runobj: RunObject, execution):
        handler = runobj.spec.handler
        self._force_handler(handler)
        tmp = mktemp('.json')
        environ['MLRUN_META_TMPFILE'] = tmp
        if self.spec.pythonpath:
            set_paths(self.spec.pythonpath)

        context = MLClientCtx.from_dict(runobj.to_dict(),
                                        rundb=self.spec.rundb,
                                        autocommit=False,
                                        tmp=tmp,
                                        host=socket.gethostname())
        global_context.set(context)
        sout, serr = exec_from_params(handler, runobj, context,
                                      self.spec.workdir)
        log_std(self._db_conn, runobj, sout, serr)
        return context.to_dict()


class LocalRuntime(BaseRuntime):
    kind = 'local'
    _is_remote = False

    def to_job(self, image=''):
        struct = self.to_dict()
        obj = KubejobRuntime.from_dict(struct)
        if image:
            obj.spec.image = image
        return obj

    @property
    def is_deployed(self):
        return True

    def _run(self, runobj: RunObject, execution):
        environ['MLRUN_EXEC_CONFIG'] = runobj.to_json()
        tmp = mktemp('.json')
        environ['MLRUN_META_TMPFILE'] = tmp
        if self.spec.rundb:
            environ['MLRUN_DBPATH'] = self.spec.rundb

        handler = runobj.spec.handler
        logger.info('starting local run: {} # {}'.format(
            self.spec.command, handler or 'main'))

        if handler:
            if self.spec.pythonpath:
                set_paths(self.spec.pythonpath)

            mod, fn = load_module(self.spec.command, handler)
            context = MLClientCtx.from_dict(runobj.to_dict(),
                                            rundb=self.spec.rundb,
                                            autocommit=False,
                                            tmp=tmp,
                                            host=socket.gethostname())
            mod.global_mlrun_context = context
            global_context.set(context)
            sout, serr = exec_from_params(fn, runobj, context,
                                          self.spec.workdir)
            log_std(self._db_conn, runobj, sout, serr, skip=self.is_child)
            return context.to_dict()

        else:
            if self.spec.mode == 'pass':
                cmd = [self.spec.command]
            else:
                cmd = [executable, self.spec.command]

            env = None
            if self.spec.pythonpath:
                pypath = self.spec.pythonpath
                if 'PYTHONPATH' in environ:
                    pypath = '{}:{}'.format(environ['PYTHONPATH'], pypath)
                env = {'PYTHONPATH': pypath}

            sout, serr = run_exec(cmd, self.spec.args, env=env,
                                  cwd=self.spec.workdir)
            log_std(self._db_conn, runobj, sout, serr, skip=self.is_child)

            try:
                with open(tmp) as fp:
                    resp = fp.read()
                remove(tmp)
                if resp:
                    return json.loads(resp)
                logger.error('empty context tmp file')
            except FileNotFoundError:
                logger.info('no context file found')
            return runobj.to_dict()


def set_paths(pythonpath=''):
    paths = pythonpath.split(':')
    if not paths:
        return
    for p in paths:
        abspath = os.path.abspath(p)
        if abspath not in sys.path:
            sys.path.append(abspath)


def load_module(file_name, handler):
    """Load module from file name"""
    path = Path(file_name)
    mod_name = path.name
    if path.suffix:
        mod_name = mod_name[:-len(path.suffix)]
    spec = imputil.spec_from_file_location(mod_name, file_name)
    if spec is None:
        raise RunError(f'cannot import from {file_name!r}')
    mod = imputil.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        fn = getattr(mod, handler)  # Will raise if name not found
    except AttributeError as e:
        raise RunError('handler {} not found in {}'.format(handler, file_name))

    return mod, fn


def run_exec(cmd, args, env=None, cwd=None):
    if args:
        cmd += args
    out = run(cmd, stdout=PIPE, stderr=PIPE, env=env, cwd=cwd)

    err = out.stderr.decode('utf-8') if out.returncode != 0 else ''
    return out.stdout.decode('utf-8'), err


def exec_from_params(handler, runobj: RunObject, context: MLClientCtx,
                     cwd=None):
    args_list = get_func_arg(handler, runobj, context)

    stdout = StringIO()
    err = ''
    val = None
    old_dir = os.getcwd()
    with redirect_stdout(stdout):
        context.set_logger_stream(stdout)
        try:
            if cwd:
                os.chdir(cwd)
            val = handler(*args_list)
            context.set_state('completed', commit=False)
        except Exception as e:
            err = str(e)
            logger.error(traceback.format_exc())
            context.set_state(error=err, commit=False)

    if cwd:
        os.chdir(old_dir)
    context.set_logger_stream(sys.stdout)
    if val:
        context.log_result('return', val)
    context.commit()
    return stdout.getvalue(), err


def get_func_arg(handler, runobj: RunObject, context: MLClientCtx):
    params = runobj.spec.parameters or {}
    inputs = runobj.spec.inputs or {}
    args_list = []
    i = 0
    args = inspect.signature(handler).parameters
    if len(args) > 0 and list(args.keys())[0] == 'context':
        args_list.append(context)
        i += 1
    if len(args) > i + 1 and list(args.keys())[i] == 'event':
        event = Event(runobj.to_dict())
        args_list.append(event)
        i += 1

    for key in list(args.keys())[i:]:
        if args[key].name in params:
            args_list.append(copy(params[key]))
        elif args[key].name in inputs:
            obj = context.get_input(key, inputs[key])
            if type(args[key].default) is str or args[key].annotation == str:
                args_list.append(obj.local())
            else:
                args_list.append(context.get_input(key, inputs[key]))
        elif args[key].default is not inspect.Parameter.empty:
            args_list.append(args[key].default)
        else:
            args_list.append(None)

    return args_list
