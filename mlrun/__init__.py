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

__version__ = '0.4.3'

from .run import get_or_create_ctx, new_function, code_to_function, import_function
from .db import get_run_db
from .model import RunTemplate, NewTask, RunObject
from .kfpops import mlrun_op
from .config import config as mlconf
from .runtimes import new_model_server
from .platforms import mount_v3io, v3io_cred
from .datastore import get_object


from os import environ


def get_version():
    return __version__


if 'IGZ_NAMESPACE_DOMAIN' in environ:
    kfp_ep = 'https://dashboard.{}/pipelines/'.format(environ['IGZ_NAMESPACE_DOMAIN'])
    environ['KF_PIPELINES_UI_ENDPOINT'] = kfp_ep