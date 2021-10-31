import base64
import json
import os

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

        os.environ["V3IO_ACCESS_KEY"] = self.v3io_access_key = "1111-2222-3333-4444"
        os.environ["V3IO_USERNAME"] = self.v3io_user = "test-user"

    def _generate_runtime(self, disable_auto_mount=False):
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

    def test_fill_credentials(self, rundb_mock):
        os.environ["MLRUN_AUTH_SESSION"] = "some-access-key"

        runtime = self._generate_runtime()
        self._execute_run(runtime)
        assert (
            runtime.metadata.credentials.access_key == os.environ["MLRUN_AUTH_SESSION"]
        )
        del os.environ["MLRUN_AUTH_SESSION"]

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
        os.environ.pop("V3IO_ACCESS_KEY", None)
        os.environ.pop("V3IO_USERNAME", None)

        pvc_params = self._setup_pvc_mount()
        pvc_params_str = base64.b64encode(json.dumps(pvc_params).encode())
        mlconf.storage.auto_mount_params = pvc_params_str

        runtime = self._generate_runtime()
        runtime.apply(mlrun.auto_mount())
        assert runtime.spec.disable_auto_mount

        self._execute_run(runtime)
        rundb_mock.assert_pvc_mount_configured(pvc_params)

        # This won't work if mount type is not pvc
        mlconf.storage.auto_mount_type = "auto"
        with pytest.raises(ValueError):
            runtime.apply(mlrun.auto_mount())
