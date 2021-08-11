import os

import pytest

from mlrun.config import config as mlconf
from mlrun.runtimes import KubejobRuntime


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

    def _generate_runtime(self):
        runtime = KubejobRuntime()
        runtime.spec.image = self.image_name
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
        mlconf.storage.auto_mount_type = "v3io_cred" if cred_only else "v3io_fuse"

        runtime = self._generate_runtime()
        self._execute_run(runtime)

        rundb_mock.assert_v3io_mount_or_creds_configured(
            self.v3io_user, self.v3io_access_key, cred_only=cred_only
        )

    def test_run_with_automount_pvc(self, rundb_mock):
        mlconf.storage.auto_mount_type = "pvc"
        pvc_params = {
            "pvc_name": "test_pvc",
            "volume_name": "test_volume",
            "volume_mount_path": "/mnt/test/path",
        }

        pvc_params_str = ",".join(
            [f"{key}={value}" for key, value in pvc_params.items()]
        )
        mlconf.storage.auto_mount_params = pvc_params_str

        runtime = self._generate_runtime()
        self._execute_run(runtime)

        rundb_mock.assert_pvc_mount_configured(pvc_params)
