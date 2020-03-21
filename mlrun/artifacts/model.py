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

from .base import Artifact


class ModelArtifact(Artifact):
    _dict_fields = Artifact._dict_fields + ['framework', 'algo', 'metrics']
    kind = 'model'

    def __init__(self, key=None, body=None, format=None, framework=None,
                 algo=None, metrics=None, target_path=None):

        super().__init__(key, body, format=format, target_path=target_path)
        self.framework = framework
        self.algo = algo
        self.metrics = metrics
