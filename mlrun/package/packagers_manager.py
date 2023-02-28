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
#
from typing import Union, List, Tuple, Dict, Any, Type
from .packager import Packager

from mlrun.artifacts import Artifact
from mlrun.datastore import DataItem, is_store_uri, store_manager


class _ManagerLabels:
    MLRUN_VERSION = "mlrun_version"
    PACKAGER = "packager"
    OBJECT_TYPE = "object_type"
    ARTIFACT_TYPE = "artifact_type"


class PackagersManager:
    def __init__(self):
        self._packagers: List[Packager] = None
        self._collect_packagers()

    def pack(self, obj: Any, log_hint: Dict[str, str]) -> Artifact:
        # Get the artifact type (if user didn't pass any, the packager will use its configured default):
        artifact_type = log_hint.pop("artifact_type", None)

        # Choose the first packager fitting to the object (packagers priority is by the order they are being added,
        # last added -> higher highest priority):
        chosen_packager: Packager = None
        for packager in self._packagers:
            if packager.is_packable(object_type=type(obj), artifact_type=artifact_type):
                # Found a packager:
                chosen_packager = packager
                break
        if chosen_packager is None:
            # Packager was not found:
            raise

        # Use the packager to pack the object:
        artifact, instructions = chosen_packager.pack(obj=obj, artifact_type=artifact_type, instructions=log_hint)

        # Prepare the manager's labels:
        pass

        # Set the instructions in the artifact's spec:
        if instructions:
            artifact.spec.package_instructions = instructions

        return artifact


    def unpack(self, data_item: DataItem, type_hint: Type) -> Any:
        # Check if the data item is based on an artifact (it may be a simple path/url data item):
        package_instructions: dict = None
        if data_item.artifact_url and is_store_uri(url=data_item.artifact_url):
            # Get the instructions:
            artifact, _ = store_manager.get_store_artifact(url=data_item.artifact_url)
            package_instructions = artifact.spec.package_instructions



    def _collect_packagers(self):
        pass
