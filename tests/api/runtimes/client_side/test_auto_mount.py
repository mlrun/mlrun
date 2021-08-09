import os
import pathlib
import sys

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from mlrun import code_to_function
from mlrun.config import config as mlconf
from mlrun.runtimes.kubejob import KubejobRuntime

from ..base import TestRuntimeBase


class TestAutoMountClientSide(TestRuntimeBase):
    def custom_setup(self):
        self.image_name = "test/image:latest"
        self.code_handler = "test_func"

        os.environ["V3IO_ACCESS_KEY"] = self.v3io_access_key = "1111-2222-3333-4444"
        os.environ["V3IO_USERNAME"] = self.v3io_user = "test-user"

    def custom_setup_after_fixtures(self):
        # auto-mount is looking at this to check if we're running on Iguazio
        mlconf.igz_version = "some_version"
        # Reset those, as tests will use various values in them
        mlconf.storage.auto_mount_type = "auto"
        mlconf.storage.auto_mount_params = {}

    def _generate_kubejob_runtime(self):
        runtime = KubejobRuntime()
        runtime.spec.image = self.image_name
        return runtime

    def _generate_nuclio_runtime(self):
        runtime = code_to_function(
            name=self.name,
            project=self.project,
            filename=self.code_filename,
            handler=self.code_handler,
            kind="nuclio",
            image=self.image_name,
            description="test function",
        )
        return runtime

    def _execute_run(self, runtime, **kwargs):
        runtime.run(
            name=self.name,
            project=self.project,
            artifact_path=self.artifact_path,
            watch=False,
        )

    @property
    def assets_path(self):
        return (
            pathlib.Path(sys.modules[self.__module__].__file__).absolute().parent.parent
            / "assets"
        )

    @pytest.mark.parametrize("cred_only", [True, False])
    def test_auto_mount_v3io(
        self, db: Session, client: TestClient, cred_only, rundb_mock
    ):
        mlconf.storage.auto_mount_type = "v3io_cred" if cred_only else "v3io_fuse"
        mlconf.storage.auto_mount_params = {}

        runtime = self._generate_kubejob_runtime()
        self._execute_run(runtime)

        rundb_mock.assert_v3io_mount_or_creds_configured(
            self.v3io_user, self.v3io_access_key, cred_only=cred_only
        )

        runtime = self._generate_nuclio_runtime()
        runtime.deploy(project=self.project)
        rundb_mock.assert_v3io_mount_or_creds_configured(self.v3io_user, self.v3io_access_key, cred_only=cred_only)

    def test_run_with_automount_pvc(self, db: Session, client: TestClient, rundb_mock):
        mlconf.storage.auto_mount_type = "pvc"
        pvc_params = {
            "pvc_name": "test_pvc",
            "volume_name": "test_volume",
            "volume_mount_path": "/mnt/test/path",
        }
        mlconf.storage.auto_mount_params = pvc_params.copy()
        runtime = self._generate_kubejob_runtime()
        self._execute_run(runtime)

        rundb_mock.assert_pvc_mount_configured(pvc_params)

        runtime = self._generate_nuclio_runtime()
        runtime.deploy(project=self.project)
        rundb_mock.assert_pvc_mount_configured(pvc_params)
