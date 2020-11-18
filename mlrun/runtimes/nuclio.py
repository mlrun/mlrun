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
import inspect
import json
import socket

from .serving import serving_subkind
from ..db import get_or_set_dburl
from ..model import RunTemplate
from ..execution import MLClientCtx
from ..serving.v1_serving import nuclio_serving_init
from ..serving.server import v2_serving_init
from .local import get_func_arg


def nuclio_init_hook(context, data, kind):
    if kind == "serving":
        nuclio_serving_init(context, data)
    elif kind == serving_subkind:
        v2_serving_init(context, data)
    elif kind in ["mlrun", "jobs"]:
        nuclio_jobs_init(context, data)
    else:
        raise ValueError("unsupported nuclio kind")


def nuclio_jobs_init(context, data):
    setattr(context, "mlrun_handler", nuclio_jobs_handler)
    setattr(context, "globals", data)


def nuclio_jobs_handler(context, event):
    paths = event.path.strip("/").split("/")

    if not paths or paths[0] not in context.globals:
        return context.Response(
            body="function name {} does not exist".format(paths[0]),
            content_type="text/plain",
            status_code=400,
        )

    fhandler = context.globals[paths[0]]
    if not inspect.isfunction(fhandler) or paths[0].startswith("_"):
        return context.Response(
            body="{} is not a public function handler".format(paths[0]),
            content_type="text/plain",
            status_code=400,
        )

    out = get_or_set_dburl()
    if out:
        context.logger.info("logging run results to: {}".format(out))

    newspec = event.body
    if newspec and not isinstance(newspec, dict):
        newspec = json.loads(newspec)

    ctx = MLClientCtx.from_dict(
        newspec,
        rundb=out,
        autocommit=False,
        log_stream=context.logger,
        host=socket.gethostname(),
    )

    args = get_func_arg(fhandler, RunTemplate.from_dict(ctx.to_dict()), ctx)
    try:
        val = fhandler(*args)
        if val:
            ctx.log_result("return", val)
    except Exception as e:
        err = str(e)
        ctx.set_state(error=err)
    return ctx.to_json()
