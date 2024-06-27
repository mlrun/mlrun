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
import mimetypes
from http import HTTPStatus

import mlrun.common.schemas
import mlrun.utils.singleton
import server.api.api.utils
import server.api.utils.auth.verifier
import server.api.utils.singletons.k8s
from mlrun import store_manager
from mlrun.errors import err_to_str
from mlrun.utils import logger


class Files(
    metaclass=mlrun.utils.singleton.Singleton,
):
    def get_filestat(
        self,
        auth_info: mlrun.common.schemas.AuthInfo,
        path: str = "",
        schema: str = None,
        user: str = None,
        secrets: dict = None,
    ):
        return self._get_filestat(schema, path, user, auth_info, secrets=secrets)

    def delete_artifact_data(
        self,
        auth_info: mlrun.common.schemas.AuthInfo,
        project: str,
        path: str = "",
        schema: str = None,
        user: str = "",
        secrets: dict = None,
    ):
        secrets = secrets or {}

        # Verify that, similar project secrets, secrets passed by users are not internal.
        # Internal secrets are internal for the use of MLRun only. If the user may pass in any arbitrary secret
        # to override its value, we allow the user to interfere with how MLRun manages its secrets internally,
        # which can lead to unexpected results.
        for secret_key in secrets.keys():
            server.api.crud.Secrets().validate_internal_project_secret_key_allowed(
                secret_key
            )

        project_secrets = self._verify_and_get_project_secrets(project)
        project_secrets.update(secrets)

        self._delete_artifact_data(
            schema, path, user, auth_info, project_secrets, project
        )

    def _get_filestat(
        self,
        schema: str,
        path: str,
        user: str,
        auth_info: mlrun.common.schemas.AuthInfo,
        secrets: dict = None,
    ):
        path = self._resolve_obj_path(schema, path, user)

        logger.debug("Got get filestat request", path=path)

        secrets = secrets or {}
        secrets.update(server.api.api.utils.get_secrets(auth_info))

        stat = None
        try:
            stat = store_manager.object(url=path, secrets=secrets).stat()
        except FileNotFoundError as exc:
            server.api.api.utils.log_and_raise(
                HTTPStatus.NOT_FOUND.value, path=path, err=err_to_str(exc)
            )

        ctype, _ = mimetypes.guess_type(path)
        if not ctype:
            ctype = "application/octet-stream"

        return {
            "size": stat.size,
            "modified": stat.modified,
            "mimetype": ctype,
        }

    def _delete_artifact_data(
        self,
        schema: str,
        path: str,
        user: str,
        auth_info: mlrun.common.schemas.AuthInfo,
        secrets: dict = None,
        project: str = "",
    ):
        path = self._resolve_obj_path(schema, path, user)

        secrets = secrets or {}
        secrets.update(server.api.api.utils.get_secrets(auth_info))

        obj = store_manager.object(url=path, secrets=secrets, project=project)
        obj.delete()

    @staticmethod
    def _resolve_obj_path(schema: str, path: str, user: str):
        path = server.api.api.utils.get_obj_path(schema, path, user=user)
        if not path:
            server.api.api.utils.log_and_raise(
                HTTPStatus.NOT_FOUND.value,
                path=path,
                err="illegal path prefix or schema",
            )
        return path

    @staticmethod
    def _verify_and_get_project_secrets(project):
        if not server.api.utils.singletons.k8s.get_k8s_helper(
            silent=True
        ).is_running_inside_kubernetes_cluster():
            return {}

        secrets_data = server.api.crud.Secrets().list_project_secrets(
            project,
            mlrun.common.schemas.SecretProviderName.kubernetes,
            allow_secrets_from_k8s=True,
        )
        return secrets_data.secrets or {}
