# Copyright 2023 Iguazio
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

import tempfile
import uuid

import pytest

from tests.system.base import TestMLRunSystem


@TestMLRunSystem.skip_test_if_env_not_configured
class TestAPIArtifacts(TestMLRunSystem):
    project_name = "db-system-test-project"

    @pytest.mark.enterprise
    def test_import_artifact(self):
        temp_dir = tempfile.mkdtemp()
        key = f"artifact_key_{uuid.uuid4()}"
        body = "my test artifact"
        artifact = self.project.log_artifact(
            key, body=body, local_path=f"{temp_dir}/test_artifact.txt"
        )
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".yaml", delete=True
        ) as temp_file:
            artifact.export(temp_file.name)
            artifact = self.project.import_artifact(
                temp_file.name, new_key=f"imported_artifact_key_{uuid.uuid4()}"
            )
        assert artifact.to_dataitem().get().decode() == body
