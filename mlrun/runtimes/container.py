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
import time
from base64 import b64encode

from ..db import RunDBError
from ..builder import build_runtime
from ..utils import get_in, logger
from .base import BaseRuntime
from .utils import add_code_metadata, default_image_name


class ContainerRuntime(BaseRuntime):
    kind = 'container'
    _is_remote = True

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

    @property
    def is_deployed(self):
        if self.spec.image:
            return True

        db = self._get_db()
        if db and db.kind == 'http':
            try:
                db.get_builder_status(self, logs=False)
            except RunDBError:
                pass

        if self.spec.image:
            return True
        if self.status.state and self.status.state == 'ready':
            return True
        return False

    def build_config(self, image='', base_image=None,
                     commands: list = None, secret=None):
        self.spec.build.image = image or self.spec.build.image
        if commands:
            if not isinstance(commands, list):
                raise ValueError('commands must be a string list')
            self.spec.build.commands = self.spec.build.commands or []
            self.spec.build.commands += commands
        if secret:
            self.spec.build.secret = secret
        if base_image:
            self.spec.build.base_image = base_image

    def deploy(self, with_mlrun=True, watch=True):

        self.spec.build.image = self.spec.build.image \
                                or default_image_name(self)
        self.spec.image = ''
        self.status.state = ''
        add_code_metadata(self.metadata.labels)

        db = self._get_db()
        if db and db.kind == 'http':
            logger.info('starting build on remote cluster, image: {}'.format(
                self.spec.build.image))
            data = db.remote_builder(self, with_mlrun)
            self.status.state = get_in(data, 'data.status.state')
            self.status.build_pod = get_in(data, 'data.status.build_pod')
            self.spec.image = get_in(data, 'data.spec.image')
            ready = data.get('ready', False)
            if watch:
                state = self._build_watch(watch)
                ready = state == 'ready'
                self.status.state = state
        else:
            ready = build_runtime(self, with_mlrun, watch)

        self._is_built = ready
        return ready

    def _build_watch(self, watch=True, logs=True):
        db = self._get_db()
        offset = 0
        try:
            text = db.get_builder_status(self, 0, logs=logs)
        except RunDBError:
            raise ValueError('function or build process not found')

        if text:
            print(text.decode())
        if watch:
            while self.status.state in ['pending', 'running']:
                offset += len(text)
                time.sleep(2)
                text = db.get_builder_status(self, offset, logs=logs)
                if text:
                    print(text.decode(), end='')

        return self.status.state

    def builder_status(self, watch=True, logs=True):
        db = self._get_db()
        if db and db.kind == 'http':
            return self._build_watch(watch, logs)

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
