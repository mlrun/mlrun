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

import mlrun.datastore


def check_artifact_parent(
    artifact_project: str,
    expected_parent_uri: str,
) -> None:
    """
    Check if the artifact's parent URI is valid and under the same project.
    :param artifact_project:     Artifact project name
    :param expected_parent_uri:  Expected parent URI of the artifact
    :raise: MLRunInvalidArgumentError if the parent URI is invalid or not under the same project
    """
    # check if the parent_uri is a valid artifact uri and it is under the same project
    if mlrun.datastore.is_store_uri(expected_parent_uri):
        project, _, _, _, _, _ = mlrun.utils.parse_artifact_uri(
            mlrun.datastore.parse_store_uri(expected_parent_uri)[1]
        )
        if project != artifact_project:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"parent_uri ({expected_parent_uri}) must be under the same project ({artifact_project})"
            )
    else:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"parent_uri ({expected_parent_uri}) must be a valid artifact URI"
        )
