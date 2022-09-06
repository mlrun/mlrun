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
from http import HTTPStatus

from deepdiff import DeepDiff
from fastapi.testclient import TestClient

import mlrun.api.schemas


def _list_and_assert_objects(
    client: TestClient, entity_name, project, query, expected_number_of_entities
):
    entity_url_name = entity_name.replace("_", "-")
    url = f"projects/{project}/{entity_url_name}"
    if query:
        url = url + f"?{query}"
    response = client.get(url)
    assert response.status_code == HTTPStatus.OK.value
    response_body = response.json()
    assert entity_name in response_body
    number_of_entities = len(response_body[entity_name])
    assert (
        number_of_entities == expected_number_of_entities
    ), f"wrong number of {entity_name} in response - {number_of_entities} instead of {expected_number_of_entities}"
    return response_body


def _list_tags_and_assert(client: TestClient, entity_name, project, expected_tags):
    entity_url_name = entity_name.replace("_", "-")
    url = f"projects/{project}/{entity_url_name}/*/tags"
    response = client.get(url)
    assert response.status_code == HTTPStatus.OK.value
    response_body = response.json()

    assert (
        DeepDiff(
            response_body["tags"],
            expected_tags,
            ignore_order=True,
            report_repetition=True,
        )
        == {}
    )


def _patch_object(
    client: TestClient,
    project_name,
    name,
    object_update,
    object_url_path,
    additive=False,
    reference="latest",
):
    patch_mode = "replace"
    if additive:
        patch_mode = "additive"
    headers = {mlrun.api.schemas.HeaderNames.patch_mode: patch_mode}
    response = client.patch(
        f"projects/{project_name}/{object_url_path}/{name}/references/{reference}",
        json=object_update,
        headers=headers,
    )
    assert response.status_code == HTTPStatus.OK.value
    response = client.get(
        f"projects/{project_name}/{object_url_path}/{name}/references/{reference}"
    )
    return response.json()


# There will be fields added (uid for example), but we don't allow any other changes
def _assert_diff_as_expected_except_for_specific_metadata(
    expected_object, actual_object, allowed_metadata_fields, expected_diff={}
):
    exclude_paths = []
    for field in allowed_metadata_fields:
        exclude_paths.append(f"root['metadata']['{field}']")
    diff = DeepDiff(
        expected_object,
        actual_object,
        ignore_order=True,
        exclude_paths=exclude_paths,
    )
    assert diff == expected_diff


def _test_partition_by_for_feature_store_objects(
    client: TestClient, object_name, project_name, count
):
    # Basic list, establishing baseline -
    # Each object should have 3 versions, tagged "older", "newer" and "newest"
    _list_and_assert_objects(client, object_name, project_name, None, count * 3)

    # Testing partition-by with desc order (newest first)
    results = _list_and_assert_objects(
        client,
        object_name,
        project_name,
        "partition-by=name&partition-sort-by=updated&rows-per-partition=1&partition-order=desc",
        count,
    )[object_name]

    for result_object in results:
        assert result_object["metadata"]["tag"] == "newest"

    # Testing partition-by with asc order (oldest first)
    results = _list_and_assert_objects(
        client,
        object_name,
        project_name,
        "partition-by=name&partition-sort-by=updated&rows-per-partition=1&partition-order=asc",
        count,
    )[object_name]

    for result_object in results:
        assert result_object["metadata"]["tag"] == "older"

    # Test more than 1 row per group.
    results = _list_and_assert_objects(
        client,
        object_name,
        project_name,
        "partition-by=name&partition-sort-by=updated&rows-per-partition=2&partition-order=desc",
        count * 2,
    )[object_name]

    for result_object in results:
        assert result_object["metadata"]["tag"] != "older"

    # Query on additional fields, to force DB joins on these tables
    results = _list_and_assert_objects(
        client,
        object_name,
        project_name,
        "entity=ticker&feature=bid&label=owner&partition-by=name&partition-sort-by=updated",
        count,
    )[object_name]
    for result_object in results:
        assert result_object["metadata"]["tag"] == "newest"

    # Some negative testing - no sort by field
    object_url_name = object_name.replace("_", "-")
    response = client.get(
        f"projects/{project_name}/{object_url_name}?partition-by=name"
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST.value
    # An invalid partition-by field - will be failed by fastapi due to schema validation.
    response = client.get(
        f"projects/{project_name}/{object_url_name}?partition-by=key&partition-sort-by=name"
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY.value
