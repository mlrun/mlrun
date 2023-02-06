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
import pytest

import mlrun
import mlrun.api.schemas
import mlrun.errors
import tests.integration.sdk_api.base


class TestExceptionHandling(tests.integration.sdk_api.base.TestMLRunIntegration):
    def test_exception_handling(self):
        """
        This test goes through different kind of errors that can happen on the server side causing different exception
        handlers to be triggered and verifies that for all of them the actual error details returned in the response and
        that the client successfully parses them and raise the right error class
        """
        mlrun.get_or_create_project("some-project", context="./")
        # log_and_raise - mlrun code uses log_and_raise (common) which raises fastapi.HTTPException because we're
        # sending a store artifact request with an invalid json body
        # This is practically verifies that log_and_raise puts the kwargs under the details
        with pytest.raises(
            mlrun.errors.MLRunBadRequestError,
            match=rf"400 Client Error: Bad Request for url: http:\/\/(.*)\/"
            rf"{mlrun.get_run_db().get_api_path_prefix()}\/artifact\/some-project\/some-uid\/some-key: details: "
            "{'reason': 'bad JSON body'}",
        ):
            mlrun.get_run_db().api_call(
                "POST",
                "artifact/some-project/some-uid/some-key",
                body="not a valid json",
            )

        # mlrun exception - mlrun code raises an mlrun exception because we're creating a project with invalid name
        # This is handled in the mlrun/api/main.py::http_status_error_handler
        invalid_project_name = "some_project"
        # Not using client class cause it does validation on client side and we want to fail on server side
        project = mlrun.api.schemas.Project(
            metadata=mlrun.api.schemas.ProjectMetadata(name=invalid_project_name)
        )
        with pytest.raises(
            mlrun.errors.MLRunBadRequestError,
            match=rf"400 Client Error: Bad Request for url: http:\/\/(.*)\/{mlrun.get_run_db().get_api_path_prefix()}"
            r"\/projects: Failed creating project some_p"
            r"roject details: 'MLRunInvalidArgumentError\(\"Field \\'project\.metadata\.name\\' is malformed"
            r"\. Does not match required pattern: (.*)\"\)'",
        ):
            mlrun.get_run_db().create_project(project)

        # fastapi exception - fastapi exception handlers return an error response on invalid deletion strategy (as a
        # defined enum allowing only certain values)
        # This is handled in the fastapi.exception_handlers.py::request_validation_exception_handler
        invalid_deletion_strategy = "some_strategy"
        with pytest.raises(
            mlrun.errors.MLRunHTTPError,
            match=r"422 Client Error: Unprocessable Entity for url: "
            rf"http:\/\/(.*)\/{mlrun.get_run_db().get_api_path_prefix()}\/projects\/some-project-name: "
            r"Failed deleting project some-project-name details: \[{'loc':"
            r" \['header', 'x-mlrun-deletion-strategy'], 'msg': \"value is not a valid enumeration member; "
            r"permitted: 'restrict', 'restricted', 'cascade', 'cascading', 'check'\", 'type': 'type_error.enum',"
            r" 'ctx': {'enum_values': \['restrict', 'restricted', 'cascade', 'cascading', 'check']}}]",
        ):
            mlrun.get_run_db().delete_project(
                "some-project-name", deletion_strategy=invalid_deletion_strategy
            )

        # lastly let's verify that a request error (failure reaching to the server) is handled nicely
        mlrun.get_run_db().base_url = "http://does-not-exist"
        with pytest.raises(
            mlrun.errors.MLRunRuntimeError,
            match=r"HTTPConnectionPool\(host='does-not-exist', port=80\): Max retries exceeded with url: "
            rf"\/{mlrun.get_run_db().get_api_path_prefix()}\/projects\/some-project \(Caused by NewConnectionError"
            r"\('<urllib3\.connection\.HTTPConnection object at (\S*)>: Failed to establish a new connection:"
            r" \[Errno (.*)'\)\): Failed retrieving project some-project",
        ):
            mlrun.get_run_db().get_project("some-project")
