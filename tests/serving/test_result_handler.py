# Copyright 2026 Iguazio
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

"""Unit and end-to-end tests for ResultHandler (output body mapping)."""

from http import HTTPMethod
from typing import cast

import nuclio_sdk
import pytest

import mlrun
from mlrun.common.schemas.serving import APIHandlerAction
from mlrun.runtimes.nuclio.serving import ServingRuntime
from mlrun.serving.endpoint_mapping import APIHandlerConfig, BodyMappings
from mlrun.serving.result_handler import ResultHandler
from mlrun.serving.server import Response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(
    path: str,
    *mappings: tuple[str, str, bool],
    method: HTTPMethod = HTTPMethod.POST,
) -> ResultHandler:
    """Build a ResultHandler with one endpoint.

    Each mapping is (source_path, destination_path, mandatory).
    """
    bm = BodyMappings()
    for src, dest, mandatory in mappings:
        bm.add_mapping(src, destination_path=dest, mandatory=mandatory)
    config = APIHandlerConfig()
    config.add_endpoint_handler(
        path, method, APIHandlerAction.ALLOW, output_body_mappings=bm
    )
    return ResultHandler(config)


# ---------------------------------------------------------------------------
# Group 1 — Endpoint matching
# ---------------------------------------------------------------------------
class TestResultHandlerEndpointMatching:
    """ResultHandler applies the mapping only when the endpoint matches."""

    def test_exact_path_match_applies_mapping(self) -> None:
        handler = _make_handler("/predict", ("$.result", "answer", False))
        response = {"result": 42, "extra": "ignored"}
        assert handler.apply(HTTPMethod.POST, "/predict", response) == {"answer": 42}

    def test_template_path_match_applies_mapping(self) -> None:
        handler = _make_handler("/predict/{model_id}", ("$.result", "answer", False))
        response = {"result": 42}
        assert handler.apply(HTTPMethod.POST, "/predict/my-model", response) == {
            "answer": 42
        }

    def test_star_path_match_applies_mapping(self) -> None:
        handler = _make_handler("/predict/*", ("$.result", "answer", False))
        response = {"result": 42}
        assert handler.apply(HTTPMethod.POST, "/predict/anything", response) == {
            "answer": 42
        }

    def test_different_http_method_does_not_apply_mapping(self) -> None:
        """Output mapping configured for POST must not affect a GET request on the same path."""
        handler = _make_handler(
            "/predict", ("$.result", "answer", False), method=HTTPMethod.POST
        )
        response = {"result": 42, "extra": "value"}
        assert handler.apply(HTTPMethod.GET, "/predict", response) is response

    _SPECIFIC_PATHS = ["/predict/1", "/predict/{model_id}"]

    @pytest.mark.parametrize("specific_path", _SPECIFIC_PATHS)
    def test_override_specific_wins_over_star_on_same_source(
        self, specific_path: str
    ) -> None:
        """Same source on both star and specific — only specific destination survives."""
        star_bm = BodyMappings()
        star_bm.add_mapping("$.result", destination_path="result_star")

        specific_bm = BodyMappings()
        specific_bm.add_mapping("$.result", destination_path="result_specific")

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/*", HTTPMethod.POST, APIHandlerAction.ALLOW, output_body_mappings=star_bm
        )
        config.add_endpoint_handler(
            specific_path,
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            output_body_mappings=specific_bm,
        )
        handler = ResultHandler(config)

        result = handler.apply(HTTPMethod.POST, "/predict/1", {"result": 99})
        assert result == {"result_specific": 99}
        assert "result_star" not in result

    @pytest.mark.parametrize("specific_path", _SPECIFIC_PATHS)
    def test_combine_star_and_specific_different_sources(
        self, specific_path: str
    ) -> None:
        """Star and specific each map a different field — both appear in the output."""
        star_bm = BodyMappings()
        star_bm.add_mapping("$.result", destination_path="result")

        specific_bm = BodyMappings()
        specific_bm.add_mapping("$.confidence", destination_path="score")

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/*", HTTPMethod.POST, APIHandlerAction.ALLOW, output_body_mappings=star_bm
        )
        config.add_endpoint_handler(
            specific_path,
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            output_body_mappings=specific_bm,
        )
        handler = ResultHandler(config)

        result = handler.apply(
            HTTPMethod.POST,
            "/predict/1",
            {"result": "cat", "confidence": 0.95, "extra": "ignored"},
        )
        assert result == {"result": "cat", "score": 0.95}


# ---------------------------------------------------------------------------
# Group 2 — Passthrough cases
# ---------------------------------------------------------------------------
class TestResultHandlerPassthrough:
    """ResultHandler returns the response unchanged when no mapping applies."""

    def test_no_output_body_mappings_configured(self) -> None:
        config = APIHandlerConfig()
        config.add_endpoint_handler("/predict", HTTPMethod.POST, APIHandlerAction.ALLOW)
        handler = ResultHandler(config)
        response = {"result": 42, "extra": "value"}
        assert handler.apply(HTTPMethod.POST, "/predict", response) is response


# ---------------------------------------------------------------------------
# Group 3 — Happy path reshape
# ---------------------------------------------------------------------------
class TestResultHandlerReshape:
    """ResultHandler correctly extracts and remaps graph response fields."""

    def test_multiple_fields_extracted_and_remapped(self) -> None:
        handler = _make_handler(
            "/predict",
            ("$.result", "answer", False),
            ("$.confidence", "score", False),
        )
        response = {"result": "cat", "confidence": 0.95, "extra": "ignored"}
        assert handler.apply(HTTPMethod.POST, "/predict", response) == {
            "answer": "cat",
            "score": 0.95,
        }

    def test_nested_source_extracted(self) -> None:
        handler = _make_handler("/predict", ("$.result.value", "answer", False))
        response = {"result": {"value": "deep"}, "other": "ignored"}
        assert handler.apply(HTTPMethod.POST, "/predict", response) == {
            "answer": "deep"
        }


# ---------------------------------------------------------------------------
# Group 4 — Missing field behavior
# ---------------------------------------------------------------------------
class TestResultHandlerMissingFields:
    """Missing fields are either None (non-mandatory) or raise an error (mandatory)."""

    def test_missing_non_mandatory_field_is_none(self) -> None:
        handler = _make_handler(
            "/predict",
            ("$.result", "answer", False),
            ("$.nonexistent", "missing_field", False),
        )
        result = handler.apply(HTTPMethod.POST, "/predict", {"result": 42})
        assert result == {"answer": 42, "missing_field": None}

    def test_missing_mandatory_field_raises(self) -> None:
        handler = _make_handler("/predict", ("$.result", "answer", True))
        with pytest.raises(
            mlrun.errors.MLRunUnprocessableEntityError,
            match="Mandatory field 'answer' not found",
        ):
            handler.apply(HTTPMethod.POST, "/predict", {"other": "value"})

    def test_full_structure_always_returned(self) -> None:
        """All declared destinations appear in the output, even if some are None."""
        handler = _make_handler(
            "/predict",
            ("$.x", "a", False),
            ("$.y", "b", False),
            ("$.z", "c", False),
        )
        result = handler.apply(HTTPMethod.POST, "/predict", {"x": 1})
        assert result == {"a": 1, "b": None, "c": None}


# ---------------------------------------------------------------------------
# Group 5 — Non-dict response handling
# ---------------------------------------------------------------------------
class TestResultHandlerNonDictResponse:
    """Non-dict responses: silently pass through unless any mapping is mandatory."""

    def test_non_dict_string_response_with_optional_mapping_passes_through(
        self,
    ) -> None:
        """Non-dict response + all-optional mappings → return unchanged."""
        handler = _make_handler("/predict", ("$.result", "answer", False))
        response = "plain string"
        assert handler.apply(HTTPMethod.POST, "/predict", response) is response

    def test_non_dict_response_with_mandatory_mapping_raises(self) -> None:
        """Non-dict response + any mandatory mapping → raise with output body wording."""
        handler = _make_handler("/predict", ("$.result", "answer", True))
        with pytest.raises(
            mlrun.errors.MLRunUnprocessableEntityError,
            match=r"Mandatory output body mappings configured but output body is not a dict",
        ):
            handler.apply(HTTPMethod.POST, "/predict", "plain string")

    def test_non_dict_response_error_includes_actual_type(self) -> None:
        """Error message includes the actual response type to aid debugging."""
        handler = _make_handler("/predict", ("$.result", "answer", True))
        with pytest.raises(
            mlrun.errors.MLRunUnprocessableEntityError,
            match=r"got list",
        ):
            handler.apply(HTTPMethod.POST, "/predict", [1, 2, 3])

    def test_missing_mandatory_field_error_wrapped_with_output_prefix(self) -> None:
        """Mandatory-field-missing errors from apply_body_map gain an output-body prefix."""
        handler = _make_handler("/predict", ("$.result", "answer", True))
        with pytest.raises(
            mlrun.errors.MLRunUnprocessableEntityError,
            match=r"Failed to process output body mapping: Mandatory field 'answer' not found in body",
        ):
            handler.apply(HTTPMethod.POST, "/predict", {"other": "value"})


# ---------------------------------------------------------------------------
# Group 6 — End-to-end via to_mock_server()
# ---------------------------------------------------------------------------
class TestResultHandlerHttpTriggerGuard:
    """End-to-end ResultHandler behavior through ``to_mock_server()``.

    Covers the http_trigger guard, full-reshape happy paths, non-dict mandatory
    422 surfacing, and path-normalization edge cases (ML-12735) where the raw
    ``event.path`` carries a query string that would otherwise defeat route
    matching.
    """

    @staticmethod
    def _make_fn(
        output_bm: BodyMappings,
        handler=None,
        method: HTTPMethod = HTTPMethod.POST,
    ) -> ServingRuntime:
        fn = cast(
            ServingRuntime,
            mlrun.new_function("test-result-handler", kind="serving"),
        )
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/predict",
            method,
            APIHandlerAction.ALLOW,
            output_body_mappings=output_bm,
        )
        fn.set_api_handler_config(config)
        graph = fn.set_topology("flow", engine="sync")
        graph.to(
            name="handler",
            handler=handler or (lambda body, **kwargs: body),
        ).respond()
        return fn

    def test_http_trigger_true_applies_exit_mapping(self) -> None:
        """to_mock_server() sets http_trigger=True by default — exit mapping must apply."""
        bm = BodyMappings()
        bm.add_mapping("$.result", destination_path="answer")

        server = self._make_fn(bm).to_mock_server()
        try:
            resp = server.test(
                "/predict",
                method="POST",
                body={"result": 42, "extra": "ignored"},
            )
            assert resp == {"answer": 42}
        finally:
            server.wait_for_completion()

    def test_http_trigger_false_skips_exit_mapping(self) -> None:
        """When http_trigger=False, the graph response is returned as-is."""
        bm = BodyMappings()
        bm.add_mapping("$.result", destination_path="answer")

        server = self._make_fn(bm).to_mock_server()
        server.http_trigger = False
        try:
            body = {"result": 42, "extra": "ignored"}
            resp = server.test("/predict", method="POST", body=body)
            assert resp == body
        finally:
            server.wait_for_completion()

    def test_e2e_full_reshape(self) -> None:
        """End-to-end: graph returns multiple fields, exit mapping reshapes to declared structure."""
        bm = BodyMappings()
        bm.add_mapping("$.prediction", destination_path="label")
        bm.add_mapping("$.confidence", destination_path="score")

        server = self._make_fn(bm).to_mock_server()
        try:
            resp = server.test(
                "/predict",
                method="POST",
                body={"prediction": "cat", "confidence": 0.97, "debug": "ignored"},
            )
            assert resp == {"label": "cat", "score": 0.97}
        finally:
            server.wait_for_completion()

    def test_e2e_missing_non_mandatory_is_none(self) -> None:
        """End-to-end: missing non-mandatory field appears as None in the response."""
        bm = BodyMappings()
        bm.add_mapping("$.prediction", destination_path="label")
        bm.add_mapping("$.confidence", destination_path="score")  # will be missing

        server = self._make_fn(bm).to_mock_server()
        try:
            resp = server.test(
                "/predict",
                method="POST",
                body={"prediction": "dog"},
            )
            assert resp == {"label": "dog", "score": None}
        finally:
            server.wait_for_completion()

    def test_e2e_non_dict_response_with_mandatory_returns_422(self) -> None:
        """End-to-end: handler returns a non-dict response with mandatory output mapping → HTTP 422."""

        def broken_handler(body, **kwargs):
            # Returns a non-dict so the output mapping can't apply at all.
            return "not-a-dict"

        bm = BodyMappings()
        bm.add_mapping("$.prediction", destination_path="label", mandatory=True)

        fn = cast(
            ServingRuntime,
            mlrun.new_function("test-result-handler-422", kind="serving"),
        )
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/predict",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            output_body_mappings=bm,
        )
        fn.set_api_handler_config(config)
        graph = fn.set_topology("flow", engine="sync")
        graph.to(name="broken", handler=broken_handler).respond()

        server = fn.to_mock_server()
        try:
            with pytest.raises(
                RuntimeError,
                match=r"failed \(422\):.*Mandatory output body mappings configured but output body is not a dict",
            ):
                server.test("/predict", method="POST", body={"any": "input"})
        finally:
            server.wait_for_completion()

    @pytest.mark.parametrize(
        "endpoint_path",
        ["/predict", "/responses/{response_id}/input_items", "/predict/*"],
        ids=["exact", "template_subpath", "star"],
    )
    def test_e2e_query_string_applies_mapping(self, endpoint_path: str) -> None:
        """Query string in event.path must not block the output BM (ML-12735).

        BM renames ``$.result → answer`` and the handler returns ``extra_field``
        which must be filtered. If route lookup is defeated by ``?query=...``,
        the response passes through unchanged and the assertion fails.
        """
        bm = BodyMappings()
        bm.add_mapping("$.result", destination_path="answer")

        fn = cast(
            ServingRuntime,
            mlrun.new_function("test-result-handler-qs", kind="serving"),
        )
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            endpoint_path,
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            output_body_mappings=bm,
        )
        fn.set_api_handler_config(config)
        graph = fn.set_topology("flow", engine="sync")
        graph.to(
            name="handler",
            handler=lambda body, **kwargs: {"result": 42, "extra_field": "ignored"},
        ).respond()

        request_path = (
            endpoint_path.replace("{response_id}", "abc").replace("*", "foo")
            + "?debug=1"
        )
        server = fn.to_mock_server()
        try:
            resp = server.test(request_path, method="POST", body={"any": "input"})
            assert resp == {"answer": 42}
        finally:
            server.wait_for_completion()

    # ML-12706 — skip output mapping on error responses.
    # Parameterized over both Response classes the fix accepts (mlrun.serving.server.Response
    # and nuclio_sdk.Response). In real Nuclio, context.Response is nuclio_sdk.Response;
    # both must be unwrapped correctly.
    @pytest.mark.parametrize(
        "response_cls",
        [Response, nuclio_sdk.Response],
        ids=["mlrun_response", "nuclio_response"],
    )
    def test_e2e_error_response_skips_output_mapping(self, response_cls) -> None:
        """Response(status_code=404) — error body passes through with original status (ML-12706)."""
        error_body = {
            "error": {
                "message": "Response with id resp_x not found",
                "type": "invalid_request_error",
            }
        }

        def error_handler(body, **kwargs):
            return response_cls(
                body=error_body,
                status_code=404,
                content_type="application/json",
            )

        bm = BodyMappings()
        bm.add_mapping("$.id", destination_path="id", mandatory=True)
        bm.add_mapping("$.object", destination_path="object", mandatory=True)

        server = self._make_fn(
            bm, handler=error_handler, method=HTTPMethod.GET
        ).to_mock_server()
        try:
            resp = server.test("/predict", method="GET", body=None, silent=True)
            # Upstream status code must reach the caller — not be masked as 422
            assert resp.status_code == 404
            # Error body must be returned verbatim — output mapping must NOT have run
            assert resp.body == error_body
        finally:
            server.wait_for_completion()

    @pytest.mark.parametrize(
        "response_cls",
        [Response, nuclio_sdk.Response],
        ids=["mlrun_response", "nuclio_response"],
    )
    def test_e2e_response_wrapper_preserves_status_code_on_success(
        self, response_cls
    ) -> None:
        """Response(status_code=200) — output mapping applies, status_code preserved (ML-12706)."""
        success_body = {"id": "resp_1", "object": "response", "extra_field": "filter"}

        def success_handler(body, **kwargs):
            return response_cls(
                body=success_body,
                status_code=200,
                content_type="application/json",
            )

        # Distinct "output_*" destination names so a "mapping ran" verdict is unambiguous —
        # if the mapping is skipped the body stays {"id": ..., "object": ..., "extra_field": ...}.
        bm = BodyMappings()
        bm.add_mapping("$.id", destination_path="output_id", mandatory=True)
        bm.add_mapping("$.object", destination_path="output_object", mandatory=True)

        server = self._make_fn(
            bm, handler=success_handler, method=HTTPMethod.GET
        ).to_mock_server()
        try:
            resp = server.test("/predict", method="GET", body=None, silent=True)
            # Explicit Response from the handler must keep the status code
            assert resp.status_code == 200
            # Output mapping ran: keys renamed to output_*, extra_field filtered
            assert resp.body == {"output_id": "resp_1", "output_object": "response"}
        finally:
            server.wait_for_completion()
