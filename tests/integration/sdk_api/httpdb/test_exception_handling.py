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
        # log_and_raise - mlrun code uses log_and_raise (common) which raises fastapi.HTTPException because we're
        # sending a get files request for a path that does not exists
        # This is practically verifies that log_and_raise puts the kwargs under the details.reason
        with pytest.raises(
            mlrun.errors.MLRunNotFoundError,
            match=fr"404 Client Error: Not Found for url: http:\/\/(.*)\/{mlrun.get_run_db().get_api_path_prefix()}"
            r"\/files\?path=file%3A%2F%2F%2Fpath%2Fdoes%2F"
            r"not%2Fexist: details: {'reason': {'path': 'file:\/\/\/path\/does\/not\/exist', 'err': \"\[Errno 2] No suc"
            r"h file or directory: '\/path\/does\/not\/exist'\"}}",
        ):
            mlrun.get_run_db().api_call(
                "GET", "files", params={"path": "file:///path/does/not/exist"}
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
            match=fr"400 Client Error: Bad Request for url: http:\/\/(.*)\/{mlrun.get_run_db().get_api_path_prefix()}"
            r"\/projects: Failed creating project some_p"
            r"roject details: {'reason': 'MLRunInvalidArgumentError\(\"Field \\'project\.metadata\.name\\' is malformed"
            r"\. Does not match required pattern: (.*)\"\)'}",
        ):
            mlrun.get_run_db().create_project(project)

        # fastapi exception - fastapi exception handlers return an error response on invalid deletion strategy (as a
        # defined enum allowing only certain values)
        # This is handled in the fastapi.exception_handlers.py::request_validation_exception_handler
        invalid_deletion_strategy = "some_strategy"
        with pytest.raises(
            mlrun.errors.MLRunHTTPError,
            match=r"422 Client Error: Unprocessable Entity for url: "
            fr"http:\/\/(.*)\/{mlrun.get_run_db().get_api_path_prefix()}\/projects\/some-project-name: "
            r"Failed deleting project some-project-name details: \[{'loc':"
            r" \['header', 'x-mlrun-deletion-strategy'], 'msg': \"value is not a valid enumeration member; "
            r"permitted: 'restrict', 'restricted', 'cascade', 'cascading', 'check'\", 'type': 'type_error.enum',"
            r" 'ctx': {'enum_values': \['restrict', 'restricted', 'cascade', 'cascading', 'check']}}]",
        ):
            mlrun.get_run_db().delete_project(
                "some-project-name", deletion_strategy=invalid_deletion_strategy
            )

        # python exception - list endpoints uses v3io client which uses python http client which throws some exception
        # (socket.gaierror) since the v3io address is empty
        # This is handled in the mlrun/api/main.py::generic_error_handler
        with pytest.raises(
            mlrun.errors.MLRunInternalServerError,
            match=r"500 Server Error: Internal Server Error for url: http:\/\/(.*)"
            fr"\/{mlrun.get_run_db().get_api_path_prefix()}\/projects\/some-project\/model-"
            r"endpoints\?start=now-1h&end=now&top-level=False: details: {\'reason\': \"ValueError\(\'Access key must be"
            r" provided in Client\(\) arguments or in the V3IO_ACCESS_KEY environment variable\'\)\"}",
        ):
            mlrun.get_run_db().list_model_endpoints(
                "some-project", access_key="some-access-key"
            )

        # lastly let's verify that a request error (failure reaching to the server) is handled nicely
        mlrun.get_run_db().base_url = "http://does-not-exist"
        with pytest.raises(
            mlrun.errors.MLRunRuntimeError,
            match=r"HTTPConnectionPool\(host='does-not-exist', port=80\): Max retries exceeded with url: "
            fr"\/{mlrun.get_run_db().get_api_path_prefix()}\/projects\/some-project \(Caused by NewConnectionError"
            r"\('<urllib3\.connection\.HTTPConnection object at (\S*)>: Failed to establish a new connection:"
            r" \[Errno (.*)'\)\): Failed retrieving project some-project",
        ):
            mlrun.get_run_db().get_project("some-project")
