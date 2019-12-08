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

from ..builder import build_runtime, remote_builder, get_remote_status
from ..utils import get_in, logger
from .base import BaseRuntime, RunError
from ..config import config


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
        self.status.state = ''
        if commands and isinstance(commands, list):
            self.spec.build.commands = self.spec.build.commands or []
            self.spec.build.commands += commands
        if secret:
            self.spec.build.secret = secret
        if base_image:
            self.spec.build.base_image = base_image

        return self._build_image(watch, with_mlrun)

    @property
    def is_deployed(self):
        if self.spec.image:
            return True
        if self.status.state and self.status.state == 'ready':
            return True
        # TODO: check in func DB if its ready
        return False

    def _build_image(self, watch=False, with_mlrun=True):

        if config.api_service:
            logger.info('starting build on remote cluster')
            data = remote_builder(self, with_mlrun)
            self.status.state = get_in(data, 'data.status.state')
            self.status.build_pod = get_in(data, 'data.status.build_pod')
            self.spec.image = get_in(data, 'data.spec.image')
            ready = data.get('ready', False)
        else:
            ready = build_runtime(self, with_mlrun, watch)

        self._is_built = ready
        return ready

    def builder_status(self, watch=True, logs=True):
        if config.api_service:
            offset = 0 if logs else -1
            meta = self.metadata
            state, text = get_remote_status(meta.name, meta.project, meta.tag, offset)
            if text:
                print('len:', len(text))
                print(text.decode())
            return state

        else:
            pod = self.status.build_pod
            if not self.status.state == 'ready' and pod:
                k8s = self._get_k8s()
                status = k8s.get_pod_status(pod)
                if logs:
                    if watch:
                        status = k8s.watch(pod)
                    else:
                        resp = k8s.logs(pod)
                        if resp:
                            print(resp.encode())

                if status == 'succeeded':
                    self.status.build_pod = None
                    self.status.state = 'ready'
                    logger.info('build completed successfully')
                    return 'ready'
                if status in ['failed', 'error']:
                    self.status.state = status
                    logger.error(' build {}, watch the build pod logs: {}'.format(status, pod))
                    return status

                logger.info('builder status is: {}, wait for it to complete'.format(status))
            return None
