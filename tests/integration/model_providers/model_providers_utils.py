# Copyright 2025 Iguazio
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
import mlrun.errors


def create_mocked_get_store_artifact(uri_to_artifact: dict):
    def mocked_get_store_artifact(uri, **kwargs):
        artifact = uri_to_artifact.get(uri)
        if not artifact:
            raise mlrun.errors.MLRunInvalidArgumentError("Artifact uri not found")
        return artifact, None

    return mocked_get_store_artifact
