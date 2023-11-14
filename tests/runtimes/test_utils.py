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
#
import tempfile

import git
import pytest

import mlrun.config as mlconfig
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
    assert (
        repo.remote("origin").url in code_metadata
    ), "code metadata should contain git info"
    assert (
        repo.head.commit.hexsha in code_metadata
    ), "commit hash should be in code metadata"


def test_add_code_metadata_stale_remote(repo):
    # simulating a malformed / stale remote that has no url attribute
    with open(f"{repo.git_dir}/config", "a") as f:
        f.write('[remote "stale"]\n')

    # origin is still there and valid, use that
    code_metadata = mlrun.runtimes.utils.add_code_metadata(repo.working_dir)
    assert (
        repo.remote("origin").url in code_metadata
    ), "code metadata should contain git info"
    assert (
        repo.head.commit.hexsha in code_metadata
    ), "commit hash should be in code metadata"

    repo.delete_remote(repo.remote("origin"))

    code_metadata = mlrun.runtimes.utils.add_code_metadata(repo.working_dir)
    assert code_metadata is None, "code metadata should be None as there is no remote"


@pytest.mark.parametrize(
    "default_registry,resolved_image_target,secret_name,default_secret_name,expected_secret_name",
    [
        # no secret name is given and image is not auto-enrich-able or known as default registry
        # do not enrich secret
        (
            None,
            "test-image",
            None,
            "default-secret-name",
            None,
        ),
        # secret name is given, so it should be used
        (
            None,
            "test-image",
            "test-secret-name",
            "default-secret-name",
            "test-secret-name",
        ),
        # auto-enrich registry name is given without secret name, use default secret name
        (
            None,
            ".test-image",
            None,
            "default-secret-name",
            "default-secret-name",
        ),
        # auto-enrich registry name is given without secret name, use default secret name
        (
            "test-registry",
            "test-registry/test-image",
            None,
            "default-secret-name",
            "default-secret-name",
        ),
        # auto enrich registry name is given with secret name, use given secret name
        (
            None,
            ".test-image",
            "test-secret-name",
            "default-secret-name",
            "test-secret-name",
        ),
        # auto enrich registry is given but not secret name and no default secret name, leave as default
        (
            None,
            ".test-image",
            None,
            None,
            None,
        ),
    ],
)
def test_resolve_function_image_secret(
    default_registry,
    resolved_image_target,
    secret_name,
    default_secret_name,
    expected_secret_name,
):
    mlconfig.config.httpdb.builder.docker_registry = default_registry
    mlconfig.config.httpdb.builder.docker_registry_secret = default_secret_name
    assert expected_secret_name == mlrun.runtimes.utils.resolve_function_image_secret(
        resolved_image_target, secret_name
    )
