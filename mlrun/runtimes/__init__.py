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

from .base import RunError, BaseRuntime  # noqa
from .local import HandlerRuntime, LocalRuntime  # noqa
from .function import RemoteRuntime, new_model_server  # noqa
from .mpijob import MpiRuntimeV1Alpha1  # noqa
from .mpijob import MpiRuntimeV1 # noqa
from .daskjob import DaskCluster, get_dask_resource  # noqa
from .kubejob import KubejobRuntime  # noqa
from .sparkjob import SparkRuntime  # noqa
from .nuclio import nuclio_init_hook
from .serving import MLModelServer


runtime_resources_map = {
    'dask': get_dask_resource()
}


runtime_dict = {'remote': RemoteRuntime,
                'nuclio': RemoteRuntime,
                'dask': DaskCluster,
                'job': KubejobRuntime,
                'mpijob': MpiRuntimeV1Alpha1,  # legacy
                'mpijob_v1': MpiRuntimeV1,
                'spark': SparkRuntime}
