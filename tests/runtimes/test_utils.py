# Copyright 2023 Iguazio
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

import os
import tempfile
import unittest.mock

import deepdiff
import git
import pytest

import mlrun.common.constants as mlrun_constants
import mlrun.common.runtimes.constants
import mlrun.errors
import mlrun.runtimes.utils


@pytest.fixture
def repo():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = git.Repo.init(tmpdir)
        repo.create_remote("origin", "git@github.com:somewhere/else.git")

        # first commit
        tempfilename = "tempfile"
        open(f"{repo.working_dir}/{tempfilename}", "wb").close()
        repo.index.add([tempfilename])
        repo.index.commit("initialcommit")

        yield repo


def test_add_code_metadata_sanity(repo):
    code_metadata = mlrun.runtimes.utils.add_code_metadata(repo.working_dir)
    assert repo.remote("origin").url in code_metadata, (
        "code metadata should contain git info"
    )
    assert repo.head.commit.hexsha in code_metadata, (
        "commit hash should be in code metadata"
    )


def test_add_code_metadata_stale_remote(repo):
    # simulating a malformed / stale remote that has no url attribute
    with open(f"{repo.git_dir}/config", "a") as f:
        f.write('[remote "stale"]\n')

    # origin is still there and valid, use that
    code_metadata = mlrun.runtimes.utils.add_code_metadata(repo.working_dir)
    assert repo.remote("origin").url in code_metadata, (
        "code metadata should contain git info"
    )
    assert repo.head.commit.hexsha in code_metadata, (
        "commit hash should be in code metadata"
    )

    repo.delete_remote(repo.remote("origin"))

    code_metadata = mlrun.runtimes.utils.add_code_metadata(repo.working_dir)
    assert code_metadata is None, "code metadata should be None as there is no remote"


@pytest.mark.parametrize(
    "labels, labels_to_enrich, expected_labels, env_vars_to_mock, owner_to_enrich",
    [
        (
            {},
            None,
            {
                mlrun_constants.MLRunInternalLabels.owner: mlrun_constants.MLRunInternalLabels.v3io_user,
            },
            None,
            None,
        ),
        (
            {},
            None,
            {mlrun_constants.MLRunInternalLabels.owner: "test_user"},
            {"LOGNAME": "test_user", "V3IO_USERNAME": ""},
            None,
        ),
        (
            {},
            {},
            {},
            {"LOGNAME": "test_user", "V3IO_USERNAME": ""},
            None,
        ),
        (
            {mlrun_constants.MLRunInternalLabels.owner: "Mahatma"},
            None,
            {
                mlrun_constants.MLRunInternalLabels.owner: "Mahatma",
            },
            None,
            None,
        ),
        (
            {
                mlrun_constants.MLRunInternalLabels.owner: "Mahatma",
                mlrun_constants.MLRunInternalLabels.v3io_user: "Gandhi",
            },
            {},
            {
                mlrun_constants.MLRunInternalLabels.owner: "Mahatma",
                mlrun_constants.MLRunInternalLabels.v3io_user: "Gandhi",
            },
            None,
            None,
        ),
        (
            {"a": "A", "b": "B"},
            {mlrun_constants.MLRunInternalLabels.owner},
            {
                "a": "A",
                "b": "B",
                mlrun_constants.MLRunInternalLabels.owner: mlrun_constants.MLRunInternalLabels.v3io_user,
            },
            None,
            None,
        ),
        (
            {"job-type": "workflow-runner"},
            None,
            {
                "job-type": "workflow-runner",
                mlrun_constants.MLRunInternalLabels.owner: "owner_user",
            },
            None,
            "owner_user",
        ),
        (
            {"job-type": "rerun-workflow-runner"},
            None,
            {
                "job-type": "rerun-workflow-runner",
                mlrun_constants.MLRunInternalLabels.owner: "owner_user",
            },
            None,
            "owner_user",
        ),
    ],
)
def test_enrich_run_labels(
    labels, labels_to_enrich, expected_labels, env_vars_to_mock, owner_to_enrich
):
    env_vars_to_mock = env_vars_to_mock or {
        "V3IO_USERNAME": mlrun_constants.MLRunInternalLabels.v3io_user,
    }
    with unittest.mock.patch.dict(
        os.environ,
        env_vars_to_mock,
    ):
        enriched_labels = mlrun.runtimes.utils.enrich_run_labels(
            labels, labels_to_enrich, owner_to_enrich=owner_to_enrich
        )
        assert (
            deepdiff.DeepDiff(
                enriched_labels,
                expected_labels,
                ignore_order=True,
            )
            == {}
        )


@pytest.mark.parametrize(
    "labels, env_vars, owner_to_enrich, expected_owner",
    [
        # No job-type, no owner_to_enrich, should use V3IO_USERNAME
        (
            {},
            {"V3IO_USERNAME": "v3io_user", "LOGNAME": "fallback_user"},
            None,
            "v3io_user",
        ),
        # No job-type, V3IO_USERNAME empty, fallback to getpass.getuser()
        (
            {},
            {"V3IO_USERNAME": "", "LOGNAME": "fallback_user"},
            None,
            "fallback_user",
        ),
        # job-type is workflow-runner, should use owner_to_enrich
        (
            {"job-type": mlrun_constants.JOB_TYPE_WORKFLOW_RUNNER},
            {"V3IO_USERNAME": "v3io_user", "LOGNAME": "fallback_user"},
            "owner_user",
            "owner_user",
        ),
        # job-type is rerun-workflow-runner, should use owner_to_enrich
        (
            {"job-type": mlrun_constants.JOB_TYPE_RERUN_WORKFLOW_RUNNER},
            {"V3IO_USERNAME": "v3io_user", "LOGNAME": "fallback_user"},
            "owner_user",
            "owner_user",
        ),
        # job-type is workflow-runner, but no owner_to_enrich, fallback to env
        (
            {"job-type": mlrun_constants.JOB_TYPE_WORKFLOW_RUNNER},
            {"V3IO_USERNAME": "v3io_user", "LOGNAME": "fallback_user"},
            None,
            "v3io_user",
        ),
    ],
)
def test_resolve_owner(labels, env_vars, owner_to_enrich, expected_owner):
    with unittest.mock.patch.dict(os.environ, env_vars, clear=True):
        with unittest.mock.patch("getpass.getuser", return_value=env_vars["LOGNAME"]):
            owner = mlrun.runtimes.utils.resolve_owner(labels, owner_to_enrich)
            assert owner == expected_owner


def test_resolve_owner_uses_ig4_token_provider_username():
    """Test that resolve_owner uses authenticated_username from IG4 token provider."""
    # Create a mock token provider with authenticated_username
    mock_token_provider = unittest.mock.MagicMock()
    mock_token_provider.authenticated_username = "ig4_authenticated_user"

    # Create a mock db with the token provider
    mock_db = unittest.mock.MagicMock()
    mock_db.token_provider = mock_token_provider

    # Clear V3IO_USERNAME to trigger IG4 fallback
    with unittest.mock.patch.dict(os.environ, {"V3IO_USERNAME": ""}, clear=True):
        with unittest.mock.patch("mlrun.get_run_db", return_value=mock_db):
            with unittest.mock.patch("getpass.getuser", return_value="local_user"):
                owner = mlrun.runtimes.utils.resolve_owner({})
                assert owner == "ig4_authenticated_user"


def test_resolve_owner_falls_back_when_token_provider_has_no_username():
    """Test that resolve_owner falls back to getpass when token provider has no username."""
    # Create a mock token provider without authenticated_username
    mock_token_provider = unittest.mock.MagicMock()
    mock_token_provider.authenticated_username = None

    # Create a mock db with the token provider
    mock_db = unittest.mock.MagicMock()
    mock_db.token_provider = mock_token_provider

    # Clear V3IO_USERNAME to trigger fallback
    with unittest.mock.patch.dict(os.environ, {"V3IO_USERNAME": ""}, clear=True):
        with unittest.mock.patch("mlrun.get_run_db", return_value=mock_db):
            with unittest.mock.patch("getpass.getuser", return_value="local_user"):
                owner = mlrun.runtimes.utils.resolve_owner({})
                assert owner == "local_user"


def test_resolve_owner_falls_back_when_no_db():
    """Test that resolve_owner falls back to getpass when no db is available."""
    # Clear V3IO_USERNAME to trigger fallback
    with unittest.mock.patch.dict(os.environ, {"V3IO_USERNAME": ""}, clear=True):
        with unittest.mock.patch("mlrun.get_run_db", return_value=None):
            with unittest.mock.patch("getpass.getuser", return_value="local_user"):
                owner = mlrun.runtimes.utils.resolve_owner({})
                assert owner == "local_user"


def test_resolve_owner_v3io_username_takes_precedence_over_ig4():
    """Test that V3IO_USERNAME takes precedence over IG4 token provider username."""
    # Create a mock token provider with authenticated_username
    mock_token_provider = unittest.mock.MagicMock()
    mock_token_provider.authenticated_username = "ig4_authenticated_user"

    # Create a mock db with the token provider
    mock_db = unittest.mock.MagicMock()
    mock_db.token_provider = mock_token_provider

    # Set V3IO_USERNAME to verify it takes precedence
    with unittest.mock.patch.dict(
        os.environ, {"V3IO_USERNAME": "v3io_user"}, clear=True
    ):
        with unittest.mock.patch("mlrun.get_run_db", return_value=mock_db):
            owner = mlrun.runtimes.utils.resolve_owner({})
            assert owner == "v3io_user"


def test_results_to_iter_status_resolution(rundb_mock):
    """
    Test that results_to_iter correctly updates the execution state based on the results provided.
    Results objects contains result of each iteration, including their parameters and status.

    The test first simulates a scenario where one of the iteration fails and is pending a retry,
    then it simulates all iterations being successful.
    """
    results = [
        {
            "spec": {"parameters": {"p1": 2, "p2": 0}},
            "status": {
                "state": "pendingRetry",
                "error": "division by zero",
                "retry_count": None,
            },
        },
        {
            "spec": {"parameters": {"p1": 2, "p2": 1}},
            "status": {"state": "completed", "results": {"multiplier": 2.0}},
        },
        {
            "spec": {"parameters": {"p1": 2, "p2": 2}},
            "status": {"state": "completed", "results": {"multiplier": 1.0}},
        },
    ]
    run = {
        "kind": "run",
        "spec": {
            "log_level": "info",
            "parameters": {"p1": 2, "p2": 0},
            "handler": "my_function",
            "outputs": [],
            "output_path": "artifacts",
            "inputs": {},
            "notifications": [],
            "retry": {"count": 2, "backoff": {"base_delay": "30 sec"}},
            "data_stores": [],
        },
    }
    run = mlrun.run.RunObject.from_dict(run)

    execution = mlrun.execution.MLClientCtx.from_dict(
        run.to_dict(),
        rundb_mock,
        autocommit=False,
        is_api=True,
        store_run=False,
    )
    # Replace execution.commit with a no-op to avoid persisting changes during test
    execution.commit = lambda: None

    mlrun.runtimes.utils.results_to_iter(results, run, execution)
    assert execution.state == mlrun.common.runtimes.constants.RunStates.pending_retry

    # delete the failed result to simulate all iterations being successful
    results = results[1:]
    mlrun.runtimes.utils.results_to_iter(results, run, execution)
    assert execution.state == mlrun.common.runtimes.constants.RunStates.completed


@pytest.mark.parametrize(
    "output_path, owner, expected_output_path",
    [
        # Basic substitution
        (
            "/data/{{run.user}}/artifacts",
            "alice",
            "/data/alice/artifacts",
        ),
        # Multiple occurrences
        (
            "/{{run.user}}/data/{{run.user}}/artifacts",
            "bob",
            "/bob/data/bob/artifacts",
        ),
        # No template in path
        (
            "/data/artifacts",
            "alice",
            "/data/artifacts",
        ),
        # Empty output_path returns as-is
        (
            "",
            "alice",
            "",
        ),
        # None output_path returns None
        (
            None,
            "alice",
            None,
        ),
        # Empty owner returns original path
        (
            "/data/{{run.user}}/artifacts",
            "",
            "/data/{{run.user}}/artifacts",
        ),
        # None owner returns original path
        (
            "/data/{{run.user}}/artifacts",
            None,
            "/data/{{run.user}}/artifacts",
        ),
        # Both None returns None
        (
            None,
            None,
            None,
        ),
    ],
)
def test_resolve_run_user_template(output_path, owner, expected_output_path):
    result = mlrun.runtimes.utils.resolve_run_user_template(output_path, owner)
    assert result == expected_output_path


@pytest.mark.parametrize(
    "image, expected_version",
    [
        (
            "mckinsey-ig4-next-gen-docker-local.jfrog.io/spark-app:4.2.0-scala2.13-java25-ubuntu-1",
            "4.2.0",
        ),
        (
            "mckinsey-ig4-next-gen-docker-local.jfrog.io/spark-app-cuda:4.2.0-scala2.13-java25-ubuntu-1",
            "4.2.0",
        ),
        (
            "mckinsey-ig4-next-gen-docker-local.jfrog.io/spark-app:3.5.6-scala2.12-java17-ubuntu-1",
            "3.5.6",
        ),
        ("iguazio/spark-app:3.5.5-b697", "3.5.5"),
        ("somereg/spark-app:4.2.0-scala2.13+jdk25", "4.2.0"),
        ("somereg/spark-app:4.2.0_scala2.13", "4.2.0"),
        ("somereg/spark-app:4.2.0-ubuntu_22.04", "4.2.0"),
        ("somereg/spark-app:4.2.0.1", "4.2.0"),
        ("mlrun/mlrun:latest", None),
        (".spark-job-default-image", None),
        (
            "localhost:5000/spark-app:4.2.0-scala2.13-java25-ubuntu-1",
            "4.2.0",
        ),
        ("localhost:5000/spark-app", None),
        ("registry:5000/spark-app@sha256:deadbeef", None),
        ("", None),
        (None, None),
    ],
)
def test_extract_spark_version_from_image(image, expected_version):
    assert (
        mlrun.runtimes.utils.extract_spark_version_from_image(image) == expected_version
    )


_SPARK4_REGISTRY_IMAGE = "mckinsey-ig4-next-gen-docker-local.jfrog.io/spark-app"
_SPARK4_TAG = "4.2.0-scala2.13-java25-ubuntu-1"
_SPARK4_IMAGE = f"{_SPARK4_REGISTRY_IMAGE}:{_SPARK4_TAG}"
_BUILT_FUNCTION_OPAQUE_IMAGE = ".sparkjob-from-github:latest"
_BUILT_FUNCTION_BASE_IMAGE = "iguazio/spark-app:3.5.5-b697"


@pytest.fixture
def spark_platform_image_config():
    original_image = mlrun.mlconf.spark_app_image
    original_tag = mlrun.mlconf.spark_app_image_tag
    mlrun.mlconf.spark_app_image = _SPARK4_REGISTRY_IMAGE
    mlrun.mlconf.spark_app_image_tag = _SPARK4_TAG
    yield
    mlrun.mlconf.spark_app_image = original_image
    mlrun.mlconf.spark_app_image_tag = original_tag


@pytest.fixture
def no_spark_platform_image_config():
    original_image = mlrun.mlconf.spark_app_image
    original_tag = mlrun.mlconf.spark_app_image_tag
    mlrun.mlconf.spark_app_image = ""
    mlrun.mlconf.spark_app_image_tag = ""
    yield
    mlrun.mlconf.spark_app_image = original_image
    mlrun.mlconf.spark_app_image_tag = original_tag


@pytest.mark.parametrize(
    "explicit_version, image, base_image, use_default_image, expected",
    [
        (
            None,
            _BUILT_FUNCTION_OPAQUE_IMAGE,
            _BUILT_FUNCTION_BASE_IMAGE,
            False,
            ("3.5.5", _BUILT_FUNCTION_BASE_IMAGE, "3.5.5"),
        ),
        (
            "3.5.5",
            _BUILT_FUNCTION_OPAQUE_IMAGE,
            _BUILT_FUNCTION_BASE_IMAGE,
            False,
            ("3.5.5", _BUILT_FUNCTION_BASE_IMAGE, "3.5.5"),
        ),
        (
            None,
            _SPARK4_IMAGE,
            None,
            False,
            ("4.2.0", _SPARK4_IMAGE, "4.2.0"),
        ),
        (
            "3.5.5",
            "mlrun/mlrun:latest",
            None,
            False,
            ("3.5.5", "mlrun/mlrun:latest", None),
        ),
        (
            None,
            None,
            None,
            True,
            ("4.2.0", _SPARK4_IMAGE, "4.2.0"),
        ),
        (
            None,
            None,
            None,
            False,
            ("4.2.0", _SPARK4_IMAGE, "4.2.0"),
        ),
    ],
)
def test_resolve_spark_version_provenance(
    spark_platform_image_config,
    explicit_version,
    image,
    base_image,
    use_default_image,
    expected,
):
    resolution = mlrun.runtimes.utils._resolve_spark_version_provenance(
        explicit_version=explicit_version,
        image=image,
        base_image=base_image,
        use_default_image=use_default_image,
    )
    assert (
        resolution.effective_version,
        resolution.provenance_image,
        resolution.provenance_version,
    ) == expected


def test_resolve_spark_version_provenance_no_source_at_all(
    no_spark_platform_image_config,
):
    resolution = mlrun.runtimes.utils._resolve_spark_version_provenance(
        explicit_version=None,
        image=None,
        base_image=None,
        use_default_image=False,
    )
    assert (
        resolution.effective_version,
        resolution.provenance_image,
        resolution.provenance_version,
    ) == (None, "", None)


def test_resolve_spark_version_explicit_wins_over_provenance():
    resolution = mlrun.runtimes.utils.resolve_spark_version(
        explicit_version="4.2.0",
        image=_SPARK4_IMAGE,
        base_image=None,
        use_default_image=False,
    )
    assert resolution.effective_version == "4.2.0"


@pytest.mark.parametrize(
    "spark3_image, spark4_image",
    [
        (_BUILT_FUNCTION_BASE_IMAGE, _SPARK4_IMAGE),
    ],
)
def test_resolve_spark_version_derives_for_spark3_and_spark4(
    spark3_image, spark4_image
):
    spark3_resolution = mlrun.runtimes.utils.resolve_spark_version(
        explicit_version=None,
        image=None,
        base_image=spark3_image,
        use_default_image=False,
    )
    assert spark3_resolution.effective_version == "3.5.5"

    spark4_resolution = mlrun.runtimes.utils.resolve_spark_version(
        explicit_version=None,
        image=spark4_image,
        base_image=None,
        use_default_image=False,
    )
    assert spark4_resolution.effective_version == "4.2.0"


def test_resolve_spark_version_opaque_without_version_fails():
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError, match="mlrun/mlrun:latest"
    ):
        mlrun.runtimes.utils.resolve_spark_version(
            explicit_version=None,
            image="mlrun/mlrun:latest",
            base_image=None,
            use_default_image=False,
        )


def test_resolve_spark_version_no_version_source_fails(
    no_spark_platform_image_config,
):
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mlrun.runtimes.utils.resolve_spark_version(
            explicit_version=None,
            image=None,
            base_image=None,
            use_default_image=False,
        )


def test_resolve_spark_version_explicit_cannot_reuse_other_major_platform_image():
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mlrun.runtimes.utils.resolve_spark_version(
            explicit_version="3.5.5",
            image=_SPARK4_IMAGE,
            base_image=None,
            use_default_image=False,
        )


@pytest.mark.parametrize(
    "explicit_version, image",
    [
        ("4", _SPARK4_IMAGE),
        ("4.x", _SPARK4_IMAGE),
        ("v4.0.0", _SPARK4_IMAGE),
        ("3.5", _BUILT_FUNCTION_BASE_IMAGE),
    ],
)
def test_resolve_spark_version_preserves_lenient_same_major_explicit_version(
    explicit_version, image
):
    resolution = mlrun.runtimes.utils.resolve_spark_version(
        explicit_version=explicit_version,
        image=image,
        base_image=None,
        use_default_image=False,
    )

    assert resolution.effective_version == explicit_version


@pytest.mark.parametrize(
    "explicit_version, image",
    [
        ("4", _BUILT_FUNCTION_BASE_IMAGE),
        ("3.5", _SPARK4_IMAGE),
        ("4.x", _BUILT_FUNCTION_BASE_IMAGE),
        ("v4.0.0", _BUILT_FUNCTION_BASE_IMAGE),
    ],
)
def test_resolve_spark_version_rejects_lenient_other_major_explicit_version(
    explicit_version, image
):
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match="does not match image",
    ):
        mlrun.runtimes.utils.resolve_spark_version(
            explicit_version=explicit_version,
            image=image,
            base_image=None,
            use_default_image=False,
        )


def test_resolve_spark_version_preserves_explicit_version_without_provenance():
    resolution = mlrun.runtimes.utils.resolve_spark_version(
        explicit_version="3.5",
        image="mlrun/mlrun:latest",
        base_image=None,
        use_default_image=False,
    )

    assert resolution.effective_version == "3.5"


def test_resolve_spark_version_rejects_unrecognizable_major_with_provenance():
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match="has no recognizable major version",
    ):
        mlrun.runtimes.utils.resolve_spark_version(
            explicit_version="spark-four",
            image=_SPARK4_IMAGE,
            base_image=None,
            use_default_image=False,
        )


def test_resolve_spark_version_error_names_configured_image_and_tag(
    spark_platform_image_config,
):
    mlrun.mlconf.spark_app_image_tag = "latest"

    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match=f"{_SPARK4_REGISTRY_IMAGE}:latest",
    ):
        mlrun.runtimes.utils.resolve_spark_version(
            explicit_version=None,
            image=None,
            base_image=None,
            use_default_image=True,
        )
