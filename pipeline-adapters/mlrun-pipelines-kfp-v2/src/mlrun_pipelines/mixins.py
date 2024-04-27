# Copyright 2024 Iguazio
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
#


class KfpAdapterMixin:
    def apply(self, modify):
        """
        Apply a modifier to the runtime which is used to change the runtimes k8s object's spec.
        Modifiers can be either KFP modifiers or MLRun modifiers (which are compatible with KFP)

        :param modify: a modifier runnable object
        :return: the runtime (self) after the modifications
        """
        raise NotImplementedError


class PipelineProviderMixin:
    def resolve_project_from_workflow_manifest(self, workflow_manifest):
        raise NotImplementedError
