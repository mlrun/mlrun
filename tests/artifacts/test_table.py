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
import pandas
import pytest

import mlrun.artifacts.dataset


# Verify TableArtifact.get_body() doesn't hang (ML-2184)
@pytest.mark.parametrize("use_dataframe", [True, False])
def test_table_artifact_get_body(use_dataframe):
    artifact = _generate_table_artifact(use_dataframe=use_dataframe)

    artifact_body = artifact.get_body()
    print("Artifact body:\n" + artifact_body)
    assert artifact_body is not None


def _generate_table_artifact(use_dataframe=True):
    if use_dataframe:
        data_frame = pandas.DataFrame({"x": [1, 2]})
    else:
        body = "just an artifact body"

    artifact = mlrun.artifacts.dataset.TableArtifact(
        df=data_frame if use_dataframe else None,
        body=body if not use_dataframe else None,
    )
    return artifact
