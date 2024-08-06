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
import sqlalchemy.orm

import mlrun.common.schemas
import server.api.crud


def test_set_function_deletion_task_id_updates_correctly(db: sqlalchemy.orm.Session):
    function_name = "test_function"
    function_tag = "latest"
    function = {
        "metadata": {"name": function_name, "tag": function_tag},
    }
    project = "test_project"
    deletion_task_id = "12345"

    server.api.crud.Functions().store_function(
        db, project=project, function=function, name=function_name, tag=function_tag
    )

    function = server.api.crud.Functions().get_function(
        db, name=function_name, project=project, tag=function_tag
    )

    result = server.api.crud.Functions().update_function(
        db_session=db,
        function=function,
        project=project,
        updates={
            "status.deletion_task_id": deletion_task_id,
        },
    )

    assert result["status"]["deletion_task_id"] == deletion_task_id


def test_update_functions_with_api_gateway_url(db: sqlalchemy.orm.Session):
    function_name = "test-function"
    function_tag = "latest"
    function = {
        "metadata": {"name": function_name, "tag": function_tag},
    }
    project = "test-project"
    gw_host = "gw.example.com"

    # save a function object to db
    server.api.crud.Functions().store_function(
        db, project=project, function=function, name=function_name, tag=function_tag
    )
    uri = mlrun.utils.generate_object_uri(project, function_name)

    # add new external invocation URL
    server.api.crud.Functions().add_function_external_invocation_url(
        db, uri, project, gw_host
    )
    updated_function = server.api.crud.Functions().get_function(
        db, project=project, name=function_name, tag=function_tag
    )
    # check that URL is there
    assert updated_function["status"]["external_invocation_urls"][0] == gw_host

    # try to add existing external invocation URL, with a slash at the end
    server.api.crud.Functions().add_function_external_invocation_url(
        db, uri, project, gw_host + "/"
    )
    updated_function = server.api.crud.Functions().get_function(
        db, project=project, name=function_name, tag=function_tag
    )
    # check that URL isn't duplicated
    assert len(updated_function["status"]["external_invocation_urls"]) == 1

    # delete URL from the list
    server.api.crud.Functions().delete_function_external_invocation_url(
        db, uri, project, gw_host
    )
    updated_function = server.api.crud.Functions().get_function(
        db, project=project, name=function_name, tag=function_tag
    )
    # check that URL was deleted
    assert len(updated_function["status"]["external_invocation_urls"]) == 0
