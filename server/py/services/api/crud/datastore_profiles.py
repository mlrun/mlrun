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

from typing import Optional

import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.utils.singleton
from mlrun.datastore.datastore_profile import DatastoreProfile as DSProfile

import framework.utils.singletons.db
import framework.utils.singletons.k8s
import services.api.crud


class DatastoreProfiles(
    metaclass=mlrun.utils.singleton.Singleton,
):
    @staticmethod
    def _in_k8s():
        k8s_helper = framework.utils.singletons.k8s.get_k8s_helper()
        return (
            k8s_helper is not None and k8s_helper.is_running_inside_kubernetes_cluster()
        )

    def _store_secret(self, project, profile_name, profile_secret_json):
        if not self._in_k8s():
            raise mlrun.errors.MLRunInvalidArgumentError(
                "MLRun is not configured with k8s, datastore profile credentials cannot be stored securely"
            )

        adjusted_secret = {
            DSProfile.generate_secret_key(profile_name, project): profile_secret_json
        }

        services.api.crud.Secrets().store_project_secrets(
            project,
            mlrun.common.schemas.SecretsData(
                provider=mlrun.common.schemas.SecretProviderName.kubernetes,
                secrets=adjusted_secret,
            ),
            allow_internal_secrets=True,
        )

    def _delete_secret(self, project, profile_name):
        if not self._in_k8s():
            raise mlrun.errors.MLRunInvalidArgumentError(
                "MLRun is not configured with k8s, datastore profile credentials cannot be deleted"
            )

        adjusted_secret = DSProfile.generate_secret_key(profile_name, project)

        services.api.crud.Secrets().delete_project_secret(
            project,
            mlrun.common.schemas.SecretsData(
                provider=mlrun.common.schemas.SecretProviderName.kubernetes,
                secret_key=adjusted_secret,
            ),
            allow_internal_secrets=True,
        )

    def store_datastore_profile(
        self,
        session: sqlalchemy.orm.Session,
        profile_name: str,
        profile_public_json: str,
        profile_secret_json: Optional[str] = None,
        project: Optional[str] = None,
    ):
        project = project or mlrun.mlconf.default_project
        framework.utils.singletons.db.get_db().store_datastore_profile(
            session, profile_name, profile_public_json, project
        )
        if profile_secret_json:
            self._store_secret(project, profile_name, profile_secret_json)

    # Returns only the public part of datastore profile information.
    # Private info is accessed via get_secret_or_env() API
    def list_datastore_profiles(
        self,
        session: sqlalchemy.orm.Session,
        project: Optional[str] = None,
    ) -> list:
        project = project or mlrun.mlconf.default_project
        return framework.utils.singletons.db.get_db().list_datastore_profiles(
            session, project
        )

    def delete_datastore_profile(
        self,
        session: sqlalchemy.orm.Session,
        profile_name: Optional[str] = None,
        project: Optional[str] = None,
    ):
        project = project or mlrun.mlconf.default_project
        # Delete public part of the secret
        framework.utils.singletons.db.get_db().delete_datastore_profile(
            session, profile_name, project
        )
        # Delete private part of the secret
        self._delete_secret(project, profile_name)

    def get_datastore_profile(
        self,
        session: sqlalchemy.orm.Session,
        profile_name: Optional[str] = None,
        project: Optional[str] = None,
    ):
        project = project or mlrun.mlconf.default_project
        return framework.utils.singletons.db.get_db().get_datastore_profile(
            session, profile_name, project
        )
