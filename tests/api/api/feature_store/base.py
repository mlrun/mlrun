from http import HTTPStatus

from deepdiff import DeepDiff
from fastapi.testclient import TestClient

import mlrun.api.schemas


def _list_and_assert_objects(
    client: TestClient, entity_name, project, query, expected_number_of_entities
):
    entity_url_name = entity_name.replace("_", "-")
    url = f"/api/projects/{project}/{entity_url_name}"
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
        f"/api/projects/{project_name}/{object_url_path}/{name}/references/{reference}",
        json=object_update,
        headers=headers,
    )
    assert response.status_code == HTTPStatus.OK.value
    response = client.get(
        f"/api/projects/{project_name}/{object_url_path}/{name}/references/{reference}"
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
        expected_object, actual_object, ignore_order=True, exclude_paths=exclude_paths,
    )
    assert diff == expected_diff


def _test_group_by_for_feature_store_objects(
    client: TestClient, object_name, project_name, count
):
    _list_and_assert_objects(client, object_name, project_name, None, count * 3)
    results = _list_and_assert_objects(
        client,
        object_name,
        project_name,
        "group-by=name&sort-by=updated&rows-per-group=1&order=desc",
        count,
    )[object_name]

    for result_object in results:
        assert result_object["metadata"]["tag"] == "newest"

    results = _list_and_assert_objects(
        client,
        object_name,
        project_name,
        "group-by=name&sort-by=updated&rows-per-group=1&order=asc",
        count,
    )[object_name]

    for result_object in results:
        assert result_object["metadata"]["tag"] == "older"

    results = _list_and_assert_objects(
        client,
        object_name,
        project_name,
        "group-by=name&sort-by=updated&rows-per-group=2&order=desc",
        count * 2,
    )[object_name]

    for result_object in results:
        assert result_object["metadata"]["tag"] != "older"

    # Query on additional fields, to force DB joins on these tables
    results = _list_and_assert_objects(
        client,
        object_name,
        project_name,
        "entity=ticker&feature=bid&label=owner&group-by=name&sort-by=updated",
        count,
    )[object_name]
    for result_object in results:
        assert result_object["metadata"]["tag"] == "newest"

    # Some negative testing
    object_url_name = object_name.replace("_", "-")
    response = client.get(
        f"/api/projects/{project_name}/{object_url_name}?group-by=name"
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST.value
    response = client.get(
        f"/api/projects/{project_name}/{object_url_name}?group-by=key&sort-by=updated"
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST.value
