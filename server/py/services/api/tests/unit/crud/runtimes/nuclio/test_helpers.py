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

import base64
import json
import unittest.mock

import pytest

import mlrun
import mlrun.common.constants as mlrun_constants

import services.api.crud.runtimes.nuclio.function
import services.api.crud.runtimes.nuclio.helpers
from services.api.tests.unit.conftest import assets_path


@pytest.fixture(autouse=True)
def _mock_code_artifact_resolution(monkeypatch):
    """Stub `mlrun.datastore.get_store_resource` for unit tests.

    `_install_store_uri_loader` resolves the store URI to validate that the
    artifact's `kind == "code"`. Unit tests don't have a real artifact DB,
    so return a fake CodeArtifact-shaped object whose `kind` attribute is
    `"code"`. Tests that need to exercise the validation failure path can
    override via their own `monkeypatch.setattr`.
    """
    fake_artifact = unittest.mock.MagicMock()
    fake_artifact.kind = "code"
    monkeypatch.setattr(
        "mlrun.datastore.get_store_resource",
        lambda *args, **kwargs: fake_artifact,
    )


def test_compiled_function_config_nuclio_golang():
    name = f"{assets_path}/training.py"
    fn = mlrun.code_to_function(
        "nuclio", filename=name, kind="nuclio", handler="my_hand"
    )
    (
        name,
        project,
        config,
    ) = services.api.crud.runtimes.nuclio.function._compile_function_config(fn)
    assert fn.kind == "remote", "kind not set, test failed"
    assert mlrun.utils.get_in(config, "spec.build.functionSourceCode"), "no source code"
    assert mlrun.utils.get_in(config, "spec.runtime").startswith("py"), (
        "runtime not set"
    )
    assert mlrun.utils.get_in(config, "spec.handler") == "training-nuclio:my_hand", (
        "wrong handler"
    )


def test_compiled_function_config_nuclio_python():
    name = f"{assets_path}/training.py"
    project = mlrun.get_or_create_project("test")
    fn = project.set_function(name, name="nuclio", kind="nuclio", handler="my_hand")
    fn.with_annotations({"something": "somewhat"})
    (
        name,
        project,
        config,
    ) = services.api.crud.runtimes.nuclio.function._compile_function_config(fn)
    assert fn.kind == "remote", "kind not set, test failed"
    assert mlrun.utils.get_in(config, "spec.build.functionSourceCode"), "no source code"
    assert mlrun.utils.get_in(config, "spec.runtime").startswith("py"), (
        "runtime not set"
    )
    assert mlrun.utils.get_in(config, "spec.handler") == "training-nuclio:my_hand", (
        "wrong handler"
    )
    assert mlrun.utils.get_in(config, "metadata.annotations.something") == "somewhat"


def test_compiled_function_config_merges_default_pod_labels(monkeypatch):
    monkeypatch.setattr(
        mlrun.mlconf,
        "default_function_pod_labels",
        base64.b64encode(json.dumps({"team": "ml", "env": "dev"}).encode()).decode(),
    )
    name = f"{assets_path}/training.py"
    fn = mlrun.code_to_function(
        "nuclio", filename=name, kind="nuclio", handler="my_hand"
    )
    fn.metadata.labels = {"env": "prod"}
    (
        _name,
        _project,
        config,
    ) = services.api.crud.runtimes.nuclio.function._compile_function_config(fn)
    labels = mlrun.utils.get_in(config, "metadata.labels")
    # service default flows through
    assert labels["team"] == "ml"
    # function label overrides service default
    assert labels["env"] == "prod"
    # mlrun/class system label always applied
    assert labels[mlrun_constants.MLRunInternalLabels.mlrun_class] == "remote"


def test_compiled_function_config_sidecar_image_enrichment():
    mlrun.mlconf.httpdb.builder.docker_registry = "docker.io"
    name = f"{assets_path}/training.py"
    fn = mlrun.code_to_function(
        "nuclio", filename=name, kind="nuclio", handler="my_hand"
    )
    fn.with_sidecar("my-sidecar", ".mlrun/mlrun")
    (
        name,
        project,
        config,
    ) = services.api.crud.runtimes.nuclio.function._compile_function_config(fn)
    assert mlrun.utils.get_in(config, "spec.sidecars"), "No sidecars"
    assert (
        mlrun.utils.get_in(config, "spec.sidecars")[0]["image"]
        == "docker.io/mlrun/mlrun:unstable"
    ), "Image not enriched"


def test_custom_scaling_metric_specs_forwarded_to_nuclio():
    name = f"{assets_path}/training.py"
    fn = mlrun.code_to_function(
        "nuclio", filename=name, kind="nuclio", handler="my_hand"
    )
    metric_specs = [
        {
            "type": "Resource",
            "resource": {
                "name": "cpu",
                "target": {"type": "AverageValue", "averageValue": "400m"},
            },
        }
    ]
    fn.spec.custom_scaling_metric_specs = metric_specs
    (
        _name,
        _project,
        config,
    ) = services.api.crud.runtimes.nuclio.function._compile_function_config(fn)
    assert mlrun.utils.get_in(config, "spec.customScalingMetricSpecs") == metric_specs


def test_custom_scaling_metric_specs_omitted_when_empty():
    """ML-11991: When custom_scaling_metric_specs is empty, the key should
    not appear in the compiled Nuclio config."""
    name = f"{assets_path}/training.py"
    fn = mlrun.code_to_function(
        "nuclio", filename=name, kind="nuclio", handler="my_hand"
    )
    (
        _name,
        _project,
        config,
    ) = services.api.crud.runtimes.nuclio.function._compile_function_config(fn)
    assert not mlrun.utils.get_in(config, "spec.customScalingMetricSpecs")


@pytest.mark.parametrize(
    "handler, expected",
    [
        (None, ("", "main:handler")),
        ("x", ("", "x:handler")),
        ("x:y", ("", "x:y")),
        ("dir#", ("dir", "main:handler")),
        ("dir#x", ("dir", "x:handler")),
        ("dir#x:y", ("dir", "x:y")),
    ],
)
def test_resolve_work_dir_and_handler(handler, expected):
    assert (
        expected
        == services.api.crud.runtimes.nuclio.helpers.resolve_work_dir_and_handler(
            handler
        )
    )


@pytest.mark.parametrize(
    "mlrun_client_version,python_version,expected_runtime",
    [
        ("1.9.0", "3.11.16", mlrun.mlconf.default_nuclio_runtime),
        ("1.8.0", "3.9.16", "python:3.9"),
        (None, None, mlrun.mlconf.default_nuclio_runtime),
        (None, "3.9.16", mlrun.mlconf.default_nuclio_runtime),
        ("1.9.0", None, mlrun.mlconf.default_nuclio_runtime),
        ("0.0.0-unstable", "3.9.16", "python:3.9"),
        ("0.0.0-unstable", "3.12.16", "python:3.12"),
        ("1.7.0", "3.12.16", "python:3.9"),
        ("1.7.0", "3.9.16", "python:3.9"),
    ],
)
def test_resolve_nuclio_runtime_python_image(
    mlrun_client_version, python_version, expected_runtime
):
    assert (
        expected_runtime
        == services.api.crud.runtimes.nuclio.helpers.resolve_nuclio_runtime_python_image(
            mlrun_client_version, python_version
        )
    )


def test_should_fetch_source_code_for_nuclio_with_store_uri():
    """_should_fetch_source_code returns True for vanilla nuclio with store:// source."""
    func = mlrun.new_function("test-func", kind="nuclio")
    func.metadata.project = "test-proj"
    func.spec.build.source = "store://artifacts/test-proj/my_code"

    assert (
        services.api.crud.runtimes.nuclio.function._should_fetch_source_code(func)
        is True
    )
    assert func.kind == mlrun.runtimes.RuntimeKinds.remote


@pytest.mark.parametrize("kind", ["nuclio", "serving"])
def test_compile_nuclio_function_with_store_source_has_init_container(
    kind, tmp_path, monkeypatch
):
    """_compile_function_config sets up an init container for vanilla
    Nuclio AND Serving when source is a store:// CodeArtifact URI.

    Both kinds share the same store:// gate in the new code path; only
    Application is special-cased ahead of it. Parametrizing here pins
    that contract.
    """
    # Isolate cwd so the project save-on-load side-effect doesn't write a
    # project.yaml into the repo root and pollute sibling tests.
    monkeypatch.chdir(tmp_path)

    source_uri = "store://artifacts/test-proj/my_code"
    func = mlrun.new_function("test-func", kind=kind)
    func.metadata.project = "test-proj"
    func.metadata.tag = "latest"
    func.spec.build.source = source_uri
    func.spec.function_handler = "main:handler"
    func.spec.image = "python:3.11"

    services.api.crud.runtimes.nuclio.function._compile_function_config(func)

    init_containers = func.spec.config.get("spec.initContainers") or []
    assert len(init_containers) == 1
    assert (
        init_containers[0]["name"]
        == mlrun.common.constants.SOURCE_LOADER_INIT_CONTAINER_NAME
    )
    assert init_containers[0]["command"] == ["mlrun", "load-source"]
    assert source_uri in init_containers[0]["args"]

    # Original store:// URI is stashed so it survives Nuclio builder seeing an empty source.
    assert func.status.application_source == source_uri

    # Nuclio's build phase needs *some* baked code (otherwise it rejects the
    # CRD with "Function must have either spec.build.path,
    # spec.build.functionSourceCode, spec.build.image or spec.image"). We
    # bake a dedicated loader module that resolves the user's real handler
    # at runtime via the MLRUN_REAL_HANDLER env var. spec.handler is
    # rewritten to point at the loader; the original is preserved in
    # status.original_handler.
    assert func.spec.build.functionSourceCode
    decoded_loader = base64.b64decode(func.spec.build.functionSourceCode).decode(
        "utf-8"
    )
    assert "MLRUN_REAL_HANDLER" in decoded_loader
    assert "import importlib" in decoded_loader
    assert mlrun.common.constants.DEFAULT_SOURCE_CODE_TARGET_DIR in decoded_loader

    # Handler rewritten to the loader.
    loader_module = mlrun.common.constants.STORE_URI_HANDLER_LOADER_MODULE
    assert func.spec.function_handler == f"{loader_module}:handler"
    # Original handler stashed for redeploy recovery.
    assert func.status.original_handler == "main:handler"
    # Loader reads user's handler from env var.
    env_list = func.spec.config.get("spec.env", [])
    real_handler_env = next(
        (e for e in env_list if e.get("name") == "MLRUN_REAL_HANDLER"), None
    )
    assert real_handler_env is not None
    assert real_handler_env["value"] == "main:handler"


@pytest.mark.parametrize("kind", ["nuclio", "serving", "application"])
def test_compile_nuclio_function_with_store_source_mounts_project_secrets_on_init_container(
    kind, tmp_path, monkeypatch
):
    """Init container for a store:// source mounts the project's K8s secret
    via envFrom, so the source-loader process can authenticate to
    credential-protected datastores (S3, GCS, Azure) and resolve DataStore
    profiles whose private members live in mlrun-project-secrets-<project>.

    All three Nuclio-derived kinds (vanilla nuclio, serving, application)
    use the same source-loader init container for store:// sources, and
    all three need the project-secret envFrom mount when the artifact's
    target_path resolves through a DataStore profile.
    """
    monkeypatch.chdir(tmp_path)

    source_uri = "store://artifacts/test-proj/my_code"
    func = mlrun.new_function("test-func", kind=kind)
    func.metadata.project = "test-proj"
    func.metadata.tag = "latest"
    func.spec.build.source = source_uri
    func.spec.function_handler = "main:handler"
    func.spec.image = "python:3.11"
    if kind == "application":
        # Application requires a sidecar (the user's app container).
        func.spec.config["spec.sidecars"] = [
            {"name": "user-app", "image": "python:3.11"}
        ]

    services.api.crud.runtimes.nuclio.function._compile_function_config(func)

    init_containers = func.spec.config.get("spec.initContainers") or []
    assert len(init_containers) == 1
    init_container = init_containers[0]
    env_from = init_container.get("envFrom") or []
    project_secret_refs = [
        e
        for e in env_from
        if e.get("secretRef", {}).get("name") == "mlrun-project-secrets-test-proj"
    ]
    assert len(project_secret_refs) == 1, (
        f"Expected exactly one project-secrets envFrom entry, got {env_from}"
    )
    # `optional: true` so the deploy doesn't fail when a project has no
    # secrets at all (the K8s secret is created lazily by set_secrets).
    assert project_secret_refs[0]["secretRef"].get("optional") is True


def test_nuclio_store_source_preserved_on_redeploy():
    """status.application_source preserves store:// URI across re-deploys.

    Simulates the post-deploy state where spec.build.source has been cleared
    (Nuclio builder must not see store://) and the original is held in
    status.application_source. _should_fetch_source_code must still detect
    it so a redeploy reconfigures the init container.
    """
    source_uri = "store://artifacts/test-proj/my_code"
    func = mlrun.new_function("test-func", kind="nuclio")
    func.metadata.project = "test-proj"
    func.status.application_source = source_uri
    func.spec.build.source = ""

    assert (
        services.api.crud.runtimes.nuclio.function._should_fetch_source_code(func)
        is True
    )


def test_compile_nuclio_function_with_store_source_redeploy_is_idempotent(
    tmp_path, monkeypatch
):
    """Recompile after a deploy must not duplicate init containers, env vars,
    loader source, or overwrite original_handler.

    Simulates the persisted state of a function that has already been deployed
    once with a store:// source: spec.build.source has been cleared,
    status.application_source / status.original_handler hold the originals,
    spec.build.functionSourceCode is the loader stub, spec.handler points at
    the loader, and both spec.config and spec.base_spec already carry an
    init-container entry (the latter is what triggered the "Duplicate value"
    K8s error before the base_spec dedupe fix).
    """
    monkeypatch.chdir(tmp_path)

    source_uri = "store://artifacts/test-proj/my_code"
    loader_module = mlrun.common.constants.STORE_URI_HANDLER_LOADER_MODULE
    loader_handler = f"{loader_module}:handler"
    init_container_name = mlrun.common.constants.SOURCE_LOADER_INIT_CONTAINER_NAME
    # Seed with bytes carrying the current stub version marker so the
    # re-bake guard treats them as up-to-date and short-circuits — pins
    # the cache-preservation behavior for current-version stubs.
    original_loader_source = base64.b64encode(
        b"# stub_version=1\n# original loader bytes"
    ).decode("utf-8")

    func = mlrun.new_function("test-func", kind="nuclio")
    func.metadata.project = "test-proj"
    func.metadata.tag = "latest"
    func.spec.image = "python:3.11"

    # Post-first-deploy state.
    func.spec.build.source = ""
    func.status.application_source = source_uri
    func.status.original_handler = "main:handler"
    func.spec.function_handler = loader_handler
    func.spec.build.functionSourceCode = original_loader_source
    func.spec.config["spec.env"] = [
        {"name": "MLRUN_REAL_HANDLER", "value": "main:handler"}
    ]
    func.spec.config["spec.initContainers"] = [
        {
            "name": init_container_name,
            "command": ["mlrun", "load-source"],
            "envFrom": [
                {
                    "secretRef": {
                        "name": "mlrun-project-secrets-test-proj",
                        "optional": True,
                    }
                }
            ],
        }
    ]
    func.spec.base_spec = {
        "spec": {
            "initContainers": [
                {"name": init_container_name, "command": ["mlrun", "load-source"]}
            ],
            # base_spec also carries over the prior deploy's env. Without a
            # base_spec env dedupe these would survive into extend_config's
            # merge and produce a duplicate alongside the freshly-set entries
            # in spec.config — same class of bug as the init-container
            # duplication, just on env vars.
            "env": [
                {"name": "MLRUN_REAL_HANDLER", "value": "stale:handler"},
                {
                    "name": "PYTHONPATH",
                    "value": "/stale/path",
                },
                {"name": "USER_VAR", "value": "keep_me"},
            ],
        }
    }

    services.api.crud.runtimes.nuclio.function._compile_function_config(func)

    # Exactly one init container in spec.config (no duplication on rebuild).
    init_containers = func.spec.config.get("spec.initContainers") or []
    loader_init = [c for c in init_containers if c.get("name") == init_container_name]
    assert len(loader_init) == 1

    # envFrom is set on the freshly-compiled init container — assert it's
    # exactly one entry (no duplication when the dedupe-replace runs over
    # an init container that already had the same envFrom from a prior deploy).
    env_from = loader_init[0].get("envFrom") or []
    project_secret_refs = [
        e
        for e in env_from
        if e.get("secretRef", {}).get("name") == "mlrun-project-secrets-test-proj"
    ]
    assert len(project_secret_refs) == 1

    # extend_config later merges nuclio_spec into base_spec; without the dedupe
    # we'd see two entries with the same name here and Nuclio would reject
    # the K8s Deployment with "Duplicate value".
    base_spec_init = mlrun.utils.get_in(func.spec.base_spec, "spec.initContainers", [])
    base_loader_count = sum(
        1 for c in base_spec_init if c.get("name") == init_container_name
    )
    assert base_loader_count == 1

    # Exactly one MLRUN_REAL_HANDLER env var with the user's original handler.
    env_list = func.spec.config.get("spec.env", [])
    real_handler_envs = [e for e in env_list if e.get("name") == "MLRUN_REAL_HANDLER"]
    assert len(real_handler_envs) == 1
    assert real_handler_envs[0]["value"] == "main:handler"

    # base_spec env dedupe: the stale MLRUN_REAL_HANDLER and PYTHONPATH that
    # carried over from the prior deploy are removed BEFORE extend_config
    # merges spec.config's freshly-set values in. Net result: exactly one
    # entry per loader-managed name (the freshly-set value), plus the
    # user-set entry untouched. Without the base_spec dedupe, extend_config's
    # append-merge would produce two entries per name and Nuclio would
    # reject the K8s Deployment with "Duplicate value".
    base_env = mlrun.utils.get_in(func.spec.base_spec, "spec.env", []) or []
    base_real_handlers = [e for e in base_env if e.get("name") == "MLRUN_REAL_HANDLER"]
    assert len(base_real_handlers) == 1
    assert base_real_handlers[0]["value"] == "main:handler"
    base_pythonpaths = [e for e in base_env if e.get("name") == "PYTHONPATH"]
    assert len(base_pythonpaths) == 1
    assert base_pythonpaths[0]["value"] != "/stale/path"
    base_user_vars = [e for e in base_env if e.get("name") == "USER_VAR"]
    assert len(base_user_vars) == 1
    assert base_user_vars[0]["value"] == "keep_me"

    # Recovery: original_handler stays anchored to the user's value, not the
    # loader handler that's now in spec.function_handler.
    assert func.status.original_handler == "main:handler"
    assert func.spec.function_handler == loader_handler

    # Loader stub is short-circuited when functionSourceCode is already set —
    # we keep the bytes from the prior deploy so Nuclio's build cache sees an
    # unchanged source and can skip rebuild.
    assert func.spec.build.functionSourceCode == original_loader_source


def test_compile_nuclio_function_with_store_source_defaults_to_main_handler(
    tmp_path, monkeypatch
):
    """Vanilla Nuclio + store:// source without an explicit handler falls
    back to mlrun's existing ``main:handler`` convention.

    Matches the same fallback applied to non-store:// nuclio functions when
    ``handler=`` isn't passed: defer to Nuclio's standard module:function
    default rather than failing at compile. Users who follow the convention
    (a ``main.py`` with a ``handler`` function in their CodeArtifact) get a
    successful deploy without specifying a handler explicitly.
    """
    monkeypatch.chdir(tmp_path)

    func = mlrun.new_function("test-func", kind="nuclio")
    func.metadata.project = "test-proj"
    func.metadata.tag = "latest"
    func.spec.build.source = "store://artifacts/test-proj/my_code"
    func.spec.image = "python:3.11"
    # No handler set.
    func.spec.function_handler = None

    services.api.crud.runtimes.nuclio.function._compile_function_config(func)

    # The default flows through to status.original_handler (preserved across
    # redeploys) and to the MLRUN_REAL_HANDLER env var the loader stub reads.
    assert func.status.original_handler == "main:handler"
    env_list = func.spec.config.get("spec.env") or []
    real_handler_env = next(
        (e for e in env_list if e.get("name") == "MLRUN_REAL_HANDLER"), None
    )
    assert real_handler_env is not None
    assert real_handler_env["value"] == "main:handler"


def test_compile_nuclio_function_with_custom_source_code_target_dir(
    tmp_path, monkeypatch
):
    """source_code_target_dir override flows through to the loader stub and volume mount.

    Users overriding the default /home/mlrun_code (e.g., to layer on top of a
    base image whose user lacks write access to /home) must see the custom
    path embedded in both the generated loader's sys.path insert and the
    init-container's output directory.
    """
    monkeypatch.chdir(tmp_path)

    custom_dir = "/opt/user_code"
    source_uri = "store://artifacts/test-proj/my_code"
    func = mlrun.new_function("test-func", kind="nuclio")
    func.metadata.project = "test-proj"
    func.metadata.tag = "latest"
    func.spec.build.source = source_uri
    func.spec.build.source_code_target_dir = custom_dir
    func.spec.function_handler = "main:handler"
    func.spec.image = "python:3.11"

    services.api.crud.runtimes.nuclio.function._compile_function_config(func)

    # Loader stub embeds the custom path.
    decoded_loader = base64.b64decode(func.spec.build.functionSourceCode).decode(
        "utf-8"
    )
    assert custom_dir in decoded_loader
    # And the default isn't accidentally also present.
    assert mlrun.common.constants.DEFAULT_SOURCE_CODE_TARGET_DIR not in decoded_loader


def test_compile_nuclio_function_with_git_source_skips_loader_branch(
    tmp_path, monkeypatch
):
    """Git sources stay on Nuclio's native code-entry-type path.

    Nuclio's builder resolves git:// directly (codeEntryType="git"); the
    store:// init-container redirect only applies to URIs Nuclio cannot
    resolve. Verify a git source doesn't accidentally trigger the loader
    stub or the source-loader init container.
    """
    monkeypatch.chdir(tmp_path)

    func = mlrun.new_function("test-func", kind="nuclio")
    func.metadata.project = "test-proj"
    func.metadata.tag = "latest"
    func.spec.build.source = "git://github.com/example/repo.git#main"
    func.spec.function_handler = "main:handler"
    func.spec.image = "python:3.11"

    services.api.crud.runtimes.nuclio.function._compile_function_config(
        func, builder_env={}
    )

    # No source-loader init container — Nuclio's native git pull handles this.
    init_containers = func.spec.config.get("spec.initContainers") or []
    assert all(
        c.get("name") != mlrun.common.constants.SOURCE_LOADER_INIT_CONTAINER_NAME
        for c in init_containers
    )
    # Handler stays as the user's value (no loader substitution).
    assert func.spec.function_handler == "main:handler"
    # No store:// URI stash since the source isn't store://.
    assert not func.status.application_source
    assert not func.status.original_handler


@pytest.mark.parametrize(
    "existing_pythonpath, expected",
    [
        # No PYTHONPATH set yet.
        (None, "/work"),
        # PYTHONPATH already set, target_dir not in it — prepend.
        ("/other/path", "/work:/other/path"),
        # PYTHONPATH already set, target_dir already in it — leave unchanged.
        ("/work", "/work"),
        # PYTHONPATH set with target_dir as one of multiple entries — leave unchanged.
        ("/before:/work:/after", "/before:/work:/after"),
    ],
)
def test_inject_main_container_pythonpath_merge_logic(existing_pythonpath, expected):
    """_inject_main_container_pythonpath must idempotently maintain PYTHONPATH.

    Repeated deploys must not produce ``/work:/work`` or accumulate stale
    entries; absent PYTHONPATH must produce a single fresh entry.
    """
    func = mlrun.new_function("test-func", kind="nuclio")
    func.metadata.project = "test-proj"
    if existing_pythonpath is not None:
        func.spec.config["spec.env"] = [
            {"name": "PYTHONPATH", "value": existing_pythonpath}
        ]

    services.api.crud.runtimes.nuclio.function._inject_main_container_pythonpath(
        function=func, target_dir="/work"
    )

    env_list = func.spec.config.get("spec.env", [])
    pythonpath_envs = [e for e in env_list if e.get("name") == "PYTHONPATH"]
    assert len(pythonpath_envs) == 1
    assert pythonpath_envs[0]["value"] == expected


def test_inject_main_container_pythonpath_preserves_user_base_spec_value():
    """A PYTHONPATH set by the user directly in base_spec must be merged
    into spec.config and dropped from base_spec, not silently lost."""
    func = mlrun.new_function("test-func", kind="nuclio")
    func.metadata.project = "test-proj"
    func.spec.base_spec = {
        "spec": {"env": [{"name": "PYTHONPATH", "value": "/opt/user_lib:/opt/extras"}]}
    }

    services.api.crud.runtimes.nuclio.function._inject_main_container_pythonpath(
        function=func, target_dir="/work"
    )

    spec_pythonpaths = [
        e for e in func.spec.config.get("spec.env", []) if e.get("name") == "PYTHONPATH"
    ]
    assert len(spec_pythonpaths) == 1
    # Managed workdir first; user-set paths follow in original order.
    assert spec_pythonpaths[0]["value"] == "/work:/opt/user_lib:/opt/extras"

    base_pythonpaths = [
        e
        for e in mlrun.utils.get_in(func.spec.base_spec, "spec.env", []) or []
        if e.get("name") == "PYTHONPATH"
    ]
    assert base_pythonpaths == []


def test_compile_application_function_with_git_source_does_not_mount_project_secrets(
    tmp_path, monkeypatch
):
    """Application git+pull_at_runtime DOES use the source-loader init
    container, but the gating is store:// only — assert that branch does
    NOT pick up the project-secrets envFrom mount, preventing a behavior
    shift for existing Application git/archive deploys.
    """
    monkeypatch.chdir(tmp_path)

    func = mlrun.new_function("test-func", kind="application")
    func.metadata.project = "test-proj"
    func.metadata.tag = "latest"
    func.spec.build.source = "git://github.com/example/repo.git#main"
    func.spec.build.load_source_on_run = True
    func.spec.function_handler = "main:handler"
    func.spec.image = "python:3.11"
    # Application requires a sidecar.
    func.spec.config["spec.sidecars"] = [{"name": "user-app", "image": "python:3.11"}]

    services.api.crud.runtimes.nuclio.function._compile_function_config(
        func, builder_env={}
    )

    init_containers = func.spec.config.get("spec.initContainers") or []
    loader_init = [
        c
        for c in init_containers
        if c.get("name") == mlrun.common.constants.SOURCE_LOADER_INIT_CONTAINER_NAME
    ]
    assert len(loader_init) == 1, (
        f"Expected source-loader init container for git+pull_at_runtime, got {init_containers}"
    )
    # Critical: git/archive init containers must NOT mount project secrets —
    # this is what keeps the change from being a behavior shift for existing
    # Application git/archive deploys.
    # Pin schema: the key is absent (None) rather than an empty list — that
    # way a future code path that adds an empty `envFrom: []` placeholder
    # surfaces here as a regression instead of passing vacuously.
    assert loader_init[0].get("envFrom") is None, (
        f"git source init container should not have envFrom (got {loader_init[0].get('envFrom')!r})"
    )


@pytest.mark.parametrize("kind", ["nuclio", "serving"])
def test_compile_nuclio_function_with_git_pull_at_runtime_raises_for_non_application(
    kind, tmp_path, monkeypatch
):
    """Vanilla Nuclio/Serving + git source + load_source_on_run is unsupported.

    The init-container redirect was designed for store:// URIs only;
    git/archive sources with pull_at_runtime belong on Application kind
    (where the sidecar handles them) or on the native Nuclio path with
    pull_at_runtime=False (so Nuclio's own builder fetches the source).
    Compile-time raise makes the unsupported combination explicit instead
    of silently producing a no-init-container deploy that fails later.
    """
    monkeypatch.chdir(tmp_path)

    func = mlrun.new_function("test-func", kind=kind)
    func.metadata.project = "test-proj"
    func.metadata.tag = "latest"
    func.spec.build.source = "git://github.com/example/repo.git#main"
    func.spec.build.load_source_on_run = True
    func.spec.function_handler = "main:handler"
    func.spec.image = "python:3.11"

    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match="supported on Application kind only",
    ):
        services.api.crud.runtimes.nuclio.function._compile_function_config(func)


def test_compile_nuclio_function_with_non_code_artifact_raises(tmp_path, monkeypatch):
    """The store URI must resolve to a CodeArtifact (kind == 'code').

    Without this boundary check, a wrong-kind artifact (model, dataset, ...)
    silently flows through to the loader and fails with a cryptic
    ``ImportError`` in the pod. Failing fast at compile time gives the user
    a message that names the actual problem.
    """
    monkeypatch.chdir(tmp_path)

    fake_model = unittest.mock.MagicMock()
    fake_model.kind = "model"
    monkeypatch.setattr(
        "mlrun.datastore.get_store_resource",
        lambda *args, **kwargs: fake_model,
    )

    func = mlrun.new_function("test-func", kind="nuclio")
    func.metadata.project = "test-proj"
    func.metadata.tag = "latest"
    func.spec.build.source = "store://artifacts/test-proj/some_model"
    func.spec.function_handler = "main:handler"
    func.spec.image = "python:3.11"

    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match="expected a code artifact",
    ):
        services.api.crud.runtimes.nuclio.function._compile_function_config(func)


def test_compile_serving_function_with_topology_skips_loader_stub(
    tmp_path, monkeypatch
):
    """Serving + store:// + no explicit handler must not install the loader
    stub.

    A serving function with `set_topology()` doesn't set
    `function.spec.function_handler` — the serving wrapper handler is wired
    later by `_set_source_code_and_handler`. Installing the loader stub
    would override the wrapper and break the topology graph. The init
    container + PYTHONPATH still need to be present so the user's serving
    code is loadable.
    """
    monkeypatch.chdir(tmp_path)

    func = mlrun.new_function("test-serving", kind="serving")
    func.metadata.project = "test-proj"
    func.metadata.tag = "latest"
    func.spec.build.source = "store://artifacts/test-proj/my_code"
    # No explicit function_handler — _set_source_code_and_handler will fill
    # in the serving wrapper later.
    func.spec.image = "python:3.11"

    services.api.crud.runtimes.nuclio.function._compile_function_config(func)

    # Init container present (the user's source still needs to be loaded).
    init_containers = func.spec.config.get("spec.initContainers") or []
    loader_inits = [
        c
        for c in init_containers
        if c.get("name") == mlrun.common.constants.SOURCE_LOADER_INIT_CONTAINER_NAME
    ]
    assert len(loader_inits) == 1

    # No loader stub baked, no handler override.
    assert not func.spec.build.functionSourceCode
    loader_handler = f"{mlrun.common.constants.STORE_URI_HANDLER_LOADER_MODULE}:handler"
    assert func.spec.function_handler != loader_handler
    # No MLRUN_REAL_HANDLER env injection — the wrapper doesn't read it.
    env_list = func.spec.config.get("spec.env", [])
    assert not any(e.get("name") == "MLRUN_REAL_HANDLER" for e in env_list)


def test_compile_nuclio_function_with_store_source_redeploy_honors_handler_update(
    tmp_path, monkeypatch
):
    """A redeploy must honor an explicit handler change in spec.

    On the first deploy `function.status.original_handler` is stashed.
    If the user later updates `function.spec.function_handler` (e.g. from
    "v1:handler" to "v2:handler") and redeploys, the new value must win
    over the stashed status.
    """
    monkeypatch.chdir(tmp_path)

    func = mlrun.new_function("test-func", kind="nuclio")
    func.metadata.project = "test-proj"
    func.metadata.tag = "latest"
    func.spec.image = "python:3.11"

    # Post-first-deploy state with stashed original_handler == "v1:handler".
    func.spec.build.source = ""
    func.status.application_source = "store://artifacts/test-proj/my_code"
    func.status.original_handler = "v1:handler"
    # User updates spec.function_handler between deploys to a fresh value.
    func.spec.function_handler = "v2:handler"

    services.api.crud.runtimes.nuclio.function._compile_function_config(func)

    # status.original_handler now reflects the spec update.
    assert func.status.original_handler == "v2:handler"
    # MLRUN_REAL_HANDLER env follows the new value.
    env_list = func.spec.config.get("spec.env", [])
    real_handler_envs = [e for e in env_list if e.get("name") == "MLRUN_REAL_HANDLER"]
    assert len(real_handler_envs) == 1
    assert real_handler_envs[0]["value"] == "v2:handler"


def test_compile_nuclio_function_rebakes_loader_stub_when_version_marker_missing(
    tmp_path, monkeypatch
):
    """Stub bytes from before the version marker was introduced must be
    re-baked on next compile. Without this, an mlrun upgrade that fixes a
    stub bug would never reach existing functions whose
    ``functionSourceCode`` was set on a prior deploy.
    """
    monkeypatch.chdir(tmp_path)

    pre_marker_bytes = base64.b64encode(b"# pre-marker stub bytes\n").decode("utf-8")

    func = mlrun.new_function("test-func", kind="nuclio")
    func.metadata.project = "test-proj"
    func.metadata.tag = "latest"
    func.spec.image = "python:3.11"
    func.spec.build.source = ""
    func.status.application_source = "store://artifacts/test-proj/my_code"
    func.status.original_handler = "main:handler"
    func.spec.function_handler = (
        f"{mlrun.common.constants.STORE_URI_HANDLER_LOADER_MODULE}:handler"
    )
    func.spec.build.functionSourceCode = pre_marker_bytes

    services.api.crud.runtimes.nuclio.function._compile_function_config(func)

    # Stub re-baked: bytes changed AND now carry the version marker.
    assert func.spec.build.functionSourceCode != pre_marker_bytes
    rebaked = base64.b64decode(func.spec.build.functionSourceCode).decode("utf-8")
    assert rebaked.startswith("# stub_version=1")


def test_compile_nuclio_function_with_missing_artifact_raises_typed_error(
    tmp_path, monkeypatch
):
    """get_store_resource returning None must surface as MLRunNotFoundError,
    not the AttributeError that an unguarded `artifact.kind` access would
    raise.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "mlrun.datastore.get_store_resource",
        lambda *args, **kwargs: None,
    )

    func = mlrun.new_function("test-func", kind="nuclio")
    func.metadata.project = "test-proj"
    func.metadata.tag = "latest"
    func.spec.build.source = "store://artifacts/test-proj/missing"
    func.spec.function_handler = "main:handler"
    func.spec.image = "python:3.11"

    with pytest.raises(
        mlrun.errors.MLRunNotFoundError,
        match="Code artifact not found",
    ):
        services.api.crud.runtimes.nuclio.function._compile_function_config(func)


def test_compile_serving_function_topology_clears_loader_shaped_state_on_toggle(
    tmp_path, monkeypatch
):
    """When a serving function previously deployed with an explicit handler
    (loader stub baked, MLRUN_REAL_HANDLER env, function_handler pointing at
    the loader) is redeployed without an explicit handler, the loader-shaped
    state must be cleared so the serving wrapper can take over cleanly.
    """
    monkeypatch.chdir(tmp_path)

    loader_handler = f"{mlrun.common.constants.STORE_URI_HANDLER_LOADER_MODULE}:handler"
    stub_with_marker = base64.b64encode(
        f"# stub_version={mlrun.common.constants.STORE_URI_LOADER_STUB_VERSION}\n"
        "# loader stub\n"
        "MLRUN_REAL_HANDLER = '...'\n".encode()
    ).decode("utf-8")

    func = mlrun.new_function("test-serving", kind="serving")
    func.metadata.project = "test-proj"
    func.metadata.tag = "latest"
    func.spec.image = "python:3.11"

    # Post-explicit-handler-deploy state.
    func.spec.build.source = ""
    func.status.application_source = "store://artifacts/test-proj/my_code"
    func.status.original_handler = "old:handler"
    # User has now removed function_handler to switch to topology — but
    # spec.function_handler is still the loader handler from the prior deploy.
    func.spec.function_handler = loader_handler
    func.spec.build.functionSourceCode = stub_with_marker
    func.spec.config["spec.env"] = [
        {"name": "MLRUN_REAL_HANDLER", "value": "old:handler"},
        {"name": "USER_VAR", "value": "keep_me"},
    ]
    func.spec.base_spec = {
        "spec": {
            "env": [
                {"name": "MLRUN_REAL_HANDLER", "value": "old:handler"},
                {"name": "USER_VAR", "value": "keep_me"},
            ]
        }
    }

    # Simulate the user removing the explicit handler. This is the "toggle
    # handler -> topology" scenario: the helper sees `spec.function_handler`
    # as the loader handler (from a prior deploy), but the user's intent
    # for the upcoming deploy is topology — so the loader-shaped state
    # gets cleared.
    func.spec.function_handler = ""

    services.api.crud.runtimes.nuclio.function._compile_function_config(func)

    # Loader-shaped state cleared.
    assert not func.spec.build.functionSourceCode
    env_list = func.spec.config.get("spec.env", [])
    assert not any(e.get("name") == "MLRUN_REAL_HANDLER" for e in env_list)
    base_env = mlrun.utils.get_in(func.spec.base_spec, "spec.env", []) or []
    assert not any(e.get("name") == "MLRUN_REAL_HANDLER" for e in base_env)

    # User-set env survives.
    assert any(e.get("name") == "USER_VAR" for e in env_list)
    assert any(e.get("name") == "USER_VAR" for e in base_env)

    # Init container still present (source still needs loading).
    init_containers = func.spec.config.get("spec.initContainers") or []
    assert any(
        c.get("name") == mlrun.common.constants.SOURCE_LOADER_INIT_CONTAINER_NAME
        for c in init_containers
    )
