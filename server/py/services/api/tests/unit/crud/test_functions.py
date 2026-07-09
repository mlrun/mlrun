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
import pytest
import sqlalchemy.orm

import mlrun.common.schemas

import services.api.crud


def test_set_function_deletion_task_id_updates_correctly(db: sqlalchemy.orm.Session):
    function_name = "test_function"
    function_tag = "latest"
    function = {
        "metadata": {"name": function_name, "tag": function_tag},
        "kind": "nuclio",
    }
    project = "test_project"
    deletion_task_id = "12345"

    services.api.crud.Functions().store_function(
        db, project=project, function=function, name=function_name, tag=function_tag
    )

    function = services.api.crud.Functions().get_function(
        db, name=function_name, project=project, tag=function_tag
    )
    kind_before_update = function["kind"]

    services.api.crud.Functions().update_function(
        db_session=db,
        function=function,
        project=project,
        updates={
            "status.deletion_task_id": deletion_task_id,
        },
    )

    function = services.api.crud.Functions().get_function(
        db, name=function_name, project=project, tag=function_tag
    )
    assert function["status"]["deletion_task_id"] == deletion_task_id
    assert function["kind"] == kind_before_update


def test_update_functions_with_api_gateway_url(db: sqlalchemy.orm.Session):
    function_name = "test-function"
    function_tag = "latest"
    function = {
        "metadata": {"name": function_name, "tag": function_tag},
    }
    project = "test-project"
    gw_host = "gw.example.com"

    # save a function object to db
    services.api.crud.Functions().store_function(
        db, project=project, function=function, name=function_name, tag=function_tag
    )
    uri = mlrun.utils.generate_object_uri(project, function_name)

    # add new external invocation URL
    services.api.crud.Functions().add_function_external_invocation_url(
        db, uri, project, gw_host
    )
    updated_function = services.api.crud.Functions().get_function(
        db, project=project, name=function_name, tag=function_tag
    )
    # check that URL is there
    assert updated_function["status"]["external_invocation_urls"][0] == gw_host

    # try to add existing external invocation URL, with a slash at the end
    services.api.crud.Functions().add_function_external_invocation_url(
        db, uri, project, gw_host + "/"
    )
    updated_function = services.api.crud.Functions().get_function(
        db, project=project, name=function_name, tag=function_tag
    )
    # check that URL isn't duplicated
    assert len(updated_function["status"]["external_invocation_urls"]) == 1

    # delete URL from the list
    services.api.crud.Functions().delete_function_external_invocation_url(
        db, uri, project, gw_host
    )
    updated_function = services.api.crud.Functions().get_function(
        db, project=project, name=function_name, tag=function_tag
    )
    # check that URL was deleted
    assert len(updated_function["status"]["external_invocation_urls"]) == 0


def test_add_api_gateway_url_syncs_address_when_empty(db: sqlalchemy.orm.Session):
    """Adding an API-gateway URL to a function whose address is unset should also
    populate status.address, so the next deploy_status poll sees no address change
    and does not trigger a spurious versioned re-store."""
    project = "test-project"
    function_name = "test-function"
    function_tag = "latest"
    gw_host = "gw.example.com"

    services.api.crud.Functions().store_function(
        db,
        project=project,
        function={"metadata": {"name": function_name, "tag": function_tag}},
        name=function_name,
        tag=function_tag,
    )
    uri = mlrun.utils.generate_object_uri(project, function_name)

    # address is empty before the API gateway URL is added
    fn = services.api.crud.Functions().get_function(
        db, project=project, name=function_name, tag=function_tag
    )
    assert fn["status"].get("address", "") == ""

    services.api.crud.Functions().add_function_external_invocation_url(
        db, uri, project, gw_host
    )
    fn = services.api.crud.Functions().get_function(
        db, project=project, name=function_name, tag=function_tag
    )
    assert fn["status"]["external_invocation_urls"] == [gw_host]
    assert fn["status"]["address"] == gw_host


def test_add_api_gateway_url_does_not_overwrite_existing_address(
    db: sqlalchemy.orm.Session,
):
    """Adding an API-gateway URL must not overwrite an address that is already set."""
    project = "test-project"
    function_name = "test-function"
    function_tag = "latest"
    existing_address = "existing.example.com"
    gw_host = "gw.example.com"

    services.api.crud.Functions().store_function(
        db,
        project=project,
        function={
            "metadata": {"name": function_name, "tag": function_tag},
            "status": {"address": existing_address},
        },
        name=function_name,
        tag=function_tag,
    )
    uri = mlrun.utils.generate_object_uri(project, function_name)

    services.api.crud.Functions().add_function_external_invocation_url(
        db, uri, project, gw_host
    )
    fn = services.api.crud.Functions().get_function(
        db, project=project, name=function_name, tag=function_tag
    )
    assert fn["status"]["external_invocation_urls"] == [gw_host]
    assert fn["status"]["address"] == existing_address


def test_delete_api_gateway_url_clears_address_when_matching(
    db: sqlalchemy.orm.Session,
):
    """Removing an API-gateway URL should clear status.address when it equals the
    removed URL, and leave it unchanged otherwise."""
    project = "test-project"
    function_name = "test-function"
    function_tag = "latest"
    gw_host = "gw.example.com"
    other_host = "other.example.com"

    services.api.crud.Functions().store_function(
        db,
        project=project,
        function={
            "metadata": {"name": function_name, "tag": function_tag},
            "status": {
                "external_invocation_urls": [gw_host, other_host],
                "address": gw_host,
            },
        },
        name=function_name,
        tag=function_tag,
    )
    uri = mlrun.utils.generate_object_uri(project, function_name)

    # removing the URL that matches address → address should be cleared
    services.api.crud.Functions().delete_function_external_invocation_url(
        db, uri, project, gw_host
    )
    fn = services.api.crud.Functions().get_function(
        db, project=project, name=function_name, tag=function_tag
    )
    assert gw_host not in fn["status"]["external_invocation_urls"]
    assert fn["status"]["address"] == ""

    # removing a URL that does NOT match address → address should be untouched
    services.api.crud.Functions().store_function(
        db,
        project=project,
        function={
            "metadata": {"name": function_name, "tag": function_tag},
            "status": {
                "external_invocation_urls": [gw_host, other_host],
                "address": gw_host,
            },
        },
        name=function_name,
        tag=function_tag,
    )
    services.api.crud.Functions().delete_function_external_invocation_url(
        db, uri, project, other_host
    )
    fn = services.api.crud.Functions().get_function(
        db, project=project, name=function_name, tag=function_tag
    )
    assert other_host not in fn["status"]["external_invocation_urls"]
    assert fn["status"]["address"] == gw_host


def test_store_and_get_function_missing_project(db: sqlalchemy.orm.Session):
    project = "some-project"
    function_name = "test-function"
    function_tag = "latest"
    function = {
        "metadata": {"name": function_name, "tag": function_tag},
    }

    # store with missing project should raise error
    with pytest.raises(mlrun.errors.MLRunMissingProjectError):
        services.api.crud.Functions().store_function(
            db, project=None, function=function, name=function_name, tag=function_tag
        )

    # store with valid project
    services.api.crud.Functions().store_function(
        db, project=project, function=function, name=function_name, tag=function_tag
    )

    # get with missing project should raise error
    with pytest.raises(mlrun.errors.MLRunMissingProjectError):
        services.api.crud.Functions().get_function(
            db, name=function_name, project=None, tag=function_tag
        )

    # list with missing project should raise error
    with pytest.raises(mlrun.errors.MLRunMissingProjectError):
        services.api.crud.Functions().list_functions(
            db, name=function_name, project=None, tag=function_tag
        )

    # get with valid project
    function = services.api.crud.Functions().get_function(
        db, name=function_name, project=project, tag=function_tag
    )
    assert function["metadata"]["name"] == function_name
    assert function["metadata"]["tag"] == function_tag
