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

# Tests for the pluggable builder seam (ML-12884): the BuilderBackend selection
# gate and the BuildRequest contract. The behaviour of the Kaniko path itself is
# the byte-identical regression anchor covered by test_builder.py; here we only
# lock the *new* seam. Buildah / config-flip / force-kaniko selection is exercised
# when the second backend lands (ML-12885), so it is intentionally not tested here.

import unittest.mock

import pytest

import mlrun
import mlrun.common.schemas
import mlrun.errors
from mlrun.config import config
from mlrun.runtimes import RuntimeKinds

import framework.utils.singletons.k8s
import services.api.utils.builder


def test_resolve_builder_backend_default_is_kaniko():
    # the shipped default: no config change routes to the Kaniko adapter
    backend = services.api.utils.builder.resolve_builder_backend(_make_build_request())
    assert isinstance(backend, services.api.utils.builder.KanikoBackend)


def test_resolve_builder_backend_unknown_raises(monkeypatch):
    # a misconfigured / not-yet-registered backend fails fast with a clear error,
    # naming the offending value - rather than silently falling back to Kaniko
    monkeypatch.setattr(config.httpdb.builder, "builder_backend", "no-such-backend")
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError, match="no-such-backend"):
        services.api.utils.builder.resolve_builder_backend(_make_build_request())


def test_build_image_build_request_carries_raw_unrouted_source(monkeypatch):
    # the D1 seam contract: build_image hands the backend the *raw* source
    # descriptor, and routing it to a build context is engine-owned (below the
    # seam). A git source is a good probe - the backend rewrites the branch
    # fragment to refs/heads while routing, so the request must still hold the
    # untouched original.
    raw_source = "git://github.com/some-org/some-repo.git#main"

    captured = {}

    def _capture_request(self, request):
        captured["request"] = request
        return unittest.mock.MagicMock()

    monkeypatch.setattr(
        services.api.utils.builder.KanikoBackend, "make_build_pod", _capture_request
    )
    monkeypatch.setattr(
        config.httpdb.builder,
        "docker_registry",
        "default.docker.registry/default-repository",
    )
    _patch_k8s_helper(monkeypatch)

    function = mlrun.new_function(
        "some-function",
        "some-project",
        "some-tag",
        image="mlrun/mlrun",
        kind=RuntimeKinds.job,
    )
    function.spec.build.source = raw_source
    services.api.utils.builder.build_runtime(
        mlrun.common.schemas.AuthInfo(),
        function,
    )

    request = captured["request"]
    assert request.source == raw_source
    # not routed: neither an empty/local context nor a branch-rewritten git ref
    assert request.source not in ("/empty", "/context", ".")
    assert "refs/heads" not in request.source


def _patch_k8s_helper(monkeypatch):
    get_k8s_helper_mock = unittest.mock.Mock()
    get_k8s_helper_mock.create_pod = unittest.mock.Mock(
        side_effect=lambda pod: (pod, "some-namespace")
    )
    get_k8s_helper_mock.get_project_secret_name = unittest.mock.Mock(
        side_effect=lambda name: "name"
    )
    get_k8s_helper_mock.get_project_secret_keys = unittest.mock.Mock(
        side_effect=lambda project, filter_internal: ["KEY"]
    )
    monkeypatch.setattr(
        framework.utils.singletons.k8s,
        "get_k8s_helper",
        lambda *args, **kwargs: get_k8s_helper_mock,
    )


def _make_build_request(**overrides) -> services.api.utils.builder.BuildRequest:
    """Build a minimal BuildRequest; override only the fields a test cares about."""
    defaults = dict(
        project="some-project",
        image_target="registry/some-image:tag",
        base_image="mlrun/mlrun",
        commands=[],
        requirements=[],
        requirements_path="",
        source="",
        inline_code=None,
        inline_path=None,
        extra=None,
        builder_env={},
        builder_env_list=[],
        project_secrets=[],
        extra_args={},
        secret_name=None,
        registry=None,
        runtime_spec=None,
        project_default_function_node_selector={},
        user_unix_id=None,
        enriched_group_id=None,
        auth_info=mlrun.common.schemas.AuthInfo(),
        name="mlrun-build",
        labels={},
        verbose=False,
    )
    defaults.update(overrides)
    return services.api.utils.builder.BuildRequest(**defaults)
