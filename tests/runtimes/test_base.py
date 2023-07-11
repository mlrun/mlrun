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
import base64
import json
import os
import shutil
import tempfile

import pytest

import mlrun.errors
from mlrun.config import config as mlconf
from mlrun.runtimes import KubejobRuntime
from mlrun.runtimes.pod import AutoMountType


class TestAutoMount:
    def setup_method(self, method):
        # set auto-mount to work as if this is an Iguazio system (otherwise it may try to mount PVC)
        mlconf.igz_version = "1.1.1"
        mlconf.storage.auto_mount_type = "auto"
        mlconf.storage.auto_mount_params = ""

        self.project = "test-project"
        self.name = "test-function"
        self.image_name = "mlrun/mlrun:latest"
        self.artifact_path = "/tmp"
        self._temp_dir = tempfile.mkdtemp()

        os.environ["V3IO_ACCESS_KEY"] = self.v3io_access_key = "1111-2222-3333-4444"
        os.environ["V3IO_USERNAME"] = self.v3io_user = "test-user"

    def teardown_method(self, test_method):
        shutil.rmtree(self._temp_dir, ignore_errors=True)

    def _generate_runtime(self, disable_auto_mount=False) -> KubejobRuntime:
        runtime = KubejobRuntime()
        runtime.spec.image = self.image_name
        runtime.spec.disable_auto_mount = disable_auto_mount
        return runtime

    def _execute_run(self, runtime):
        runtime.run(
            name=self.name,
            project=self.project,
            artifact_path=self.artifact_path,
            watch=False,
        )

    @pytest.mark.parametrize("cred_only", [True, False])
    def test_auto_mount_v3io(self, cred_only, rundb_mock):
        mlconf.storage.auto_mount_type = (
            "v3io_credentials" if cred_only else "v3io_fuse"
        )

        runtime = self._generate_runtime()
        self._execute_run(runtime)

        rundb_mock.assert_v3io_mount_or_creds_configured(
            self.v3io_user, self.v3io_access_key, cred_only=cred_only
        )

        # Check that disable-auto-mount works. Need a fresh runtime, to reset its mount-applied indication.
        rundb_mock.reset()
        runtime = self._generate_runtime(disable_auto_mount=True)
        self._execute_run(runtime)
        rundb_mock.assert_no_mount_or_creds_configured()

    @pytest.mark.parametrize(
        "requirements,encoded_requirements",
        [
            # strip spaces
            (["pandas==1.0.0", "numpy==1.0.0 "], ["pandas==1.0.0", "numpy==1.0.0"]),
            # handle ranges
            (["pandas>=1.0.0, <2"], ["pandas>=1.0.0, <2"]),
            (["pandas>=1.0.0,<2"], ["pandas>=1.0.0,<2"]),
            # handle flags
            (["-r somewhere/requirements.txt"], ["-r somewhere/requirements.txt"]),
            # handle flags and specific
            # handle escaping within specific
            (
                ["-r somewhere/requirements.txt", "pandas>=1.0.0, <2"],
                ["-r somewhere/requirements.txt", "pandas>=1.0.0, <2"],
            ),
            # handle from git
            (
                ["something @ git+https://somewhere.com/a/b.git@v0.0.0#egg=something"],
                ["something @ git+https://somewhere.com/a/b.git@v0.0.0#egg=something"],
            ),
            # handle comments
            (["# dont care", "faker"], ["faker"]),
            (["faker # inline dontcare"], ["faker"]),
            (["faker #inline dontcare2"], ["faker"]),
            (
                [
                    "numpy==1.0.0 ",
                    "pandas>=1.0.0, <2",
                    "# dont care",
                    "pandas2>=1.0.0,<2 # just an inline comment",
                    "-r somewhere/requirements.txt",
                    "something @ git+https://somewhere.com/a/b.git@v0.0.0#egg=something",
                ],
                [
                    "numpy==1.0.0",
                    "pandas>=1.0.0, <2",
                    "pandas2>=1.0.0,<2",
                    "-r somewhere/requirements.txt",
                    "something @ git+https://somewhere.com/a/b.git@v0.0.0#egg=something",
                ],
            ),
        ],
    )
    def test_resolve_requirements(self, requirements, encoded_requirements):
        for requirements_as_file in [True, False]:
            if requirements_as_file:

                # create a temporary file with the requirements
                requirements = self._create_temp_requirements_file(requirements)

            encoded = self._generate_runtime().spec.build._resolve_requirements(
                requirements
            )
            assert (
                encoded == encoded_requirements
            ), f"Failed to encode {requirements} as file {requirements_as_file}"

    @pytest.mark.parametrize(
        "requirements,requirements_in_file,encoded_requirements,requirements_as_file",
        [
            (
                ["pandas==1.0.0", "numpy==1.0.0"],
                ["something==1.0.0", "otherthing==1.0.0"],
                [
                    "something==1.0.0",
                    "otherthing==1.0.0",
                    "pandas==1.0.0",
                    "numpy==1.0.0",
                ],
                False,
            ),
            (
                ["pandas==1.0.0", "numpy==1.0.0"],
                ["something==1.0.0", "otherthing==1.0.0"],
                [
                    "something==1.0.0",
                    "otherthing==1.0.0",
                    "pandas==1.0.0",
                    "numpy==1.0.0",
                ],
                True,
            ),
        ],
    )
    def test_resolve_requirements_file(
        self,
        requirements,
        requirements_in_file,
        encoded_requirements,
        requirements_as_file,
    ):
        # create requirements file
        requirements_file = self._create_temp_requirements_file(requirements_in_file)

        if requirements_as_file:
            requirements = self._create_temp_requirements_file(requirements)

        encoded = self._generate_runtime().spec.build._resolve_requirements(
            requirements, requirements_file
        )
        assert (
            encoded == encoded_requirements
        ), f"Failed to encode {requirements.extend(requirements_in_file)} as file {requirements_file}"

    def test_fill_credentials(self, rundb_mock):
        """
        expects to set the generate access key so that the API will enrich with the auth session that is being passed
        through the request headers
        """
        os.environ[
            mlrun.runtimes.constants.FunctionEnvironmentVariables.auth_session
        ] = "some-access-key"

        runtime = self._generate_runtime()
        self._execute_run(runtime)
        assert (
            runtime.metadata.credentials.access_key
            == mlrun.model.Credentials.generate_access_key
        )

    def test_auto_mount_invalid_value(self):
        # When invalid value is used, we explode
        mlconf.storage.auto_mount_type = "something_wrong"
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            auto_mount_type = AutoMountType(mlconf.storage.auto_mount_type)

        # When it's missing, we just use auto
        mlconf.storage.auto_mount_type = None
        auto_mount_type = AutoMountType(mlconf.storage.auto_mount_type)
        assert auto_mount_type == AutoMountType.auto

    @staticmethod
    def _setup_pvc_mount():
        mlconf.storage.auto_mount_type = "pvc"
        return {
            "pvc_name": "test_pvc",
            "volume_name": "test_volume",
            "volume_mount_path": "/mnt/test/path",
        }

    def test_run_with_automount_pvc(self, rundb_mock):
        pvc_params = self._setup_pvc_mount()
        # Verify that extra parameters get filtered out
        pvc_params["invalid_param"] = "blublu"

        # Try with a simple string
        pvc_params_str = ",".join(
            [f"{key}={value}" for key, value in pvc_params.items()]
        )
        mlconf.storage.auto_mount_params = pvc_params_str

        runtime = self._generate_runtime()
        self._execute_run(runtime)

        rundb_mock.assert_pvc_mount_configured(pvc_params)

        rundb_mock.reset()

        # Try with a base64 json dictionary
        pvc_params_str = base64.b64encode(json.dumps(pvc_params).encode())
        mlconf.storage.auto_mount_params = pvc_params_str

        runtime = self._generate_runtime()
        self._execute_run(runtime)

        rundb_mock.assert_pvc_mount_configured(pvc_params)

        # Try with disable-auto-mount
        rundb_mock.reset()
        runtime = self._generate_runtime(disable_auto_mount=True)
        self._execute_run(runtime)
        rundb_mock.assert_no_mount_or_creds_configured()

        # Try something that does not translate to a dictionary
        bad_params_str = base64.b64encode(
            json.dumps(["I'm", "not", "a", "dictionary"]).encode()
        )
        mlconf.storage.auto_mount_params = bad_params_str

        with pytest.raises(TypeError):
            mlconf.get_storage_auto_mount_params()

    def test_auto_mount_function_with_pvc_config(self, rundb_mock):
        pvc_params = self._setup_pvc_mount()
        pvc_params_str = base64.b64encode(json.dumps(pvc_params).encode())
        mlconf.storage.auto_mount_params = pvc_params_str

        runtime = self._generate_runtime()
        runtime.apply(mlrun.auto_mount())
        assert runtime.spec.disable_auto_mount

        self._execute_run(runtime)
        rundb_mock.assert_pvc_mount_configured(pvc_params)

        os.environ.pop("V3IO_ACCESS_KEY", None)
        # This won't work if mount type is not pvc
        mlconf.storage.auto_mount_type = "auto"
        with pytest.raises(
            ValueError, match="failed to auto mount, need to set env vars"
        ):
            runtime.apply(mlrun.auto_mount())

    @staticmethod
    def _setup_s3_mount(use_secret, non_anonymous):
        mlconf.storage.auto_mount_type = "s3"
        if use_secret:
            params = {
                "secret_name": "s3_secret",
            }
        else:
            params = {
                "aws_access_key": "some_key",
                "aws_secret_key": "some_secret_key",
            }
        if non_anonymous:
            params["non_anonymous"] = True
        return params

    @pytest.mark.parametrize("use_secret", [True, False])
    @pytest.mark.parametrize("non_anonymous", [True, False])
    def test_auto_mount_s3(self, use_secret, non_anonymous, rundb_mock):
        s3_params = self._setup_s3_mount(use_secret, non_anonymous)
        mlconf.storage.auto_mount_params = ",".join(
            [f"{key}={value}" for key, value in s3_params.items()]
        )
        runtime = self._generate_runtime()
        self._execute_run(runtime)
        rundb_mock.assert_s3_mount_configured(s3_params)

    def test_auto_mount_env(self, rundb_mock):
        expected_env = {
            "VAR1": "value1",
            "some_var_2": "some_value2",
            "one-more": "one more!!!",
        }

        mlconf.storage.auto_mount_type = "env"
        # Pass key=value pairs to the function
        mlconf.storage.auto_mount_params = ",".join(
            [f"{key}={value}" for key, value in expected_env.items()]
        )
        print(f"Auto mount params: {mlconf.storage.auto_mount_params}")
        runtime = self._generate_runtime()
        self._execute_run(runtime)
        rundb_mock.assert_env_variables(expected_env)

        # Try with a base64 json dictionary
        pvc_params_str = base64.b64encode(json.dumps(expected_env).encode())
        mlconf.storage.auto_mount_params = pvc_params_str
        print(f"Auto mount params: {mlconf.storage.auto_mount_params}")
        runtime = self._generate_runtime()
        rundb_mock.reset()
        self._execute_run(runtime)
        rundb_mock.assert_env_variables(expected_env)

    def _create_temp_requirements_file(self, requirements):
        with tempfile.NamedTemporaryFile(
            delete=False, dir=self._temp_dir, suffix=".txt"
        ) as temp_file:
            with open(temp_file.name, "w") as f:
                for requirement in requirements:
                    f.write(requirement + "\n")
            return temp_file.name
