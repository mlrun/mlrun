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

from copy import deepcopy
from ..model import RunObject
from ..utils import update_in, logger
from .base import MLRuntime, RunError


class NuclioDeployRuntime(MLRuntime):
    kind = 'nuclio'

    def _run(self, runobj: RunObject):

        from nuclio.deploy import deploy_config

        runtime = runobj.spec.runtime
        shortspec = deepcopy(runobj)
        shortspec.spec.runtime = None
        extra_env = [{'name': 'MLRUN_EXEC_CONFIG', 'value': shortspec.to_json()}]
        if self.rundb:
            extra_env.append({'name': 'MLRUN_META_DBPATH', 'value': self.rundb})

        update_in(runtime.spec, 'env', extra_env, append=True)

        uid = runobj.metadata.uid
        update_in(runtime.metadata.labels, 'mlrun/class', self.kind)
        update_in(runtime.metadata.labels, 'mlrun/uid', uid)
        if runobj.metadata.name:
            runtime.metadata.name = runobj.metadata.name
        name = runtime.metadata.name
        project = runobj.metadata.project

        if not project or not name:
            raise RunError('name and a project must be specified in the run to deploy this function')

        config = runtime.to_dict(['kind', 'apiVersion', 'spec', 'metadata'])
        print(config)

        addr = deploy_config(config, runtime.command, name, project, create_new=True)
        logger.info('function address is {}'.format(addr))
        return None

