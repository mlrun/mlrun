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

import mlrun.utils.singleton
import server.py.services.api.api.utils
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

    def _get_filestat(
        self,
        schema: str,
        path: str,
        user: str,
        auth_info: mlrun.common.schemas.AuthInfo,
        secrets: dict = None,
    ):
        _, filename = path.split(path)

        path = server.py.services.api.api.utils.get_obj_path(schema, path, user=user)
        if not path:
            server.py.services.api.api.utils.log_and_raise(
                HTTPStatus.NOT_FOUND.value,
                path=path,
                err="illegal path prefix or schema",
            )
        logger.debug("Got get filestat request", path=path)

        secrets = secrets or {}
        secrets.update(server.py.services.api.api.utils.get_secrets(auth_info))

        stat = None
        try:
            stat = store_manager.object(url=path, secrets=secrets).stat()
        except FileNotFoundError as exc:
            server.py.services.api.api.utils.log_and_raise(
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
