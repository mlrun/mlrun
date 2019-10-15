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

from base64 import b64encode

from ..builder import build_runtime
from ..utils import get_in, logger
from .base import BaseRuntime, RunError


class ContainerRuntime(BaseRuntime):
    kind = 'container'

    def with_code(self, from_file='', body=None):
        if (not body and not from_file) or (from_file and from_file.endswith('.ipynb')):
            from nuclio import build_file
            name, spec, code = build_file(from_file)
            self.spec.build.functionSourceCode = get_in(spec, 'spec.build.functionSourceCode')
            return self

        if from_file:
            with open(from_file) as fp:
                body = fp.read()
        self.spec.build.functionSourceCode = b64encode(body.encode('utf-8')).decode('utf-8')
        return self

    def build(self, image, base_image=None, commands: list = None,
              secret=None, with_mlrun=True, watch=True):
        self.spec.build.image = image
        self.spec.image = ''
        if commands and isinstance(commands, list):
            self.spec.build.commands = self.spec.build.commands or []
            self.spec.build.commands += commands
        if secret:
            self.spec.build.secret = secret
        if base_image:
            self.spec.build.base_image = base_image
        ready = self._build_image(watch, with_mlrun)
        return self

    def _build_image(self, watch=False, with_mlrun=True, execution=None):
        build = self.spec.build
        pod = build.build_pod
        if not self._is_built and pod:
            k8s = self._get_k8s()
            status = k8s.get_pod_status(pod)
            if status == 'succeeded':
                build.build_pod = None
                self._is_built = True
                logger.info('build completed successfully')
                return True
            if status in ['failed', 'error']:
                raise RunError(' build {}, watch the build pod logs: {}'.format(status, pod))
            logger.info('builder status is: {}, wait for it to complete'.format(status))
            return False

        if not build.commands and self.spec.mode == 'pass' and not build.source:
            if not self.spec.image and not build.base_image:
                raise RunError('image or base_image must be specified')
            self.spec.image = self.spec.image or build.base_image
        if self.spec.image:
            self._is_built = True
            return True

        if execution:
            execution.set_state('build')
        ready = build_runtime(self, with_mlrun, watch)
        self._is_built = ready
