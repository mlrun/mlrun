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
from os.path import expanduser

from ..config import config as mlconf
from .helpers import logger


class AzureVaultStore:
    def __init__(self, vault_name):
        try:
            from azure.identity import EnvironmentCredential
            from azure.keyvault.secrets import SecretClient
        except ImportError as exc:
            raise ImportError(
                "Azure key-vault libraries not installed, run pip install mlrun[azure-key-vault]"
            ) from exc

        self._vault_name = vault_name
        self._url = mlconf.secret_stores.azure_vault.url.format(name=vault_name)
        self._client = None

        tenant_id = self._get_secret_file_contents("tenant_id")
        client_id = self._get_secret_file_contents("client_id")

        if not tenant_id or not client_id:
            logger.info(
                "both tenant_id and client_id must be configured to use Azure vault"
            )
            return

        client_secret = self._get_secret_file_contents("secret")
        if client_secret:
            os.environ["AZURE_CLIENT_SECRET"] = client_secret

        if "AZURE_CLIENT_SECRET" not in os.environ:
            logger.info("Azure client secret could not be found")
            return

        # Azure EnvironmentCredential uses these environment variables. Populate them
        os.environ["AZURE_TENANT_ID"] = tenant_id
        os.environ["AZURE_CLIENT_ID"] = client_id

        credential = EnvironmentCredential()
        self._client = SecretClient(vault_url=self._url, credential=credential)

    @staticmethod
    def _get_secret_file_contents(file_name):
        full_path = expanduser(
            mlconf.secret_stores.azure_vault.secret_path + "/" + file_name
        )
        if os.path.isfile(full_path):
            with open(full_path) as secret_file:
                contents = secret_file.read()
            return contents
        return None

    def get_secrets(self, keys):
        # We're not checking this import, since azure-core is automatically installed by other
        # libs. Assuming we passed the checks on __init__, this is expected to work.
        from azure.core.exceptions import HttpResponseError, ResourceNotFoundError

        secrets = {}
        if not self._client:
            return secrets

        if len(keys) == 0:
            secret_details = self._client.list_properties_of_secrets()
            keys = [secret.name for secret in secret_details if secret.enabled]

        for secret in keys:
            try:
                secrets[secret] = self._client.get_secret(secret).value
            except ResourceNotFoundError:
                logger.warning(f"Secret '{secret}' is not available in Azure key vault")
            except HttpResponseError as exc:
                logger.warning(f"Exception retrieving secret '{secret}': {exc.error}")
        return secrets
