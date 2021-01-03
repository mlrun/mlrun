# Copyright 2020 Iguazio
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

import requests
import os
from os.path import expanduser
import json
from .helpers import logger
from ..config import config as mlconf
from ..k8s_utils import get_k8s_helper

vault_default_prefix = "v1/secret/data"
vault_url_env_var = "MLRUN_VAULT_URL"
vault_token_env_var = "MLRUN_VAULT_TOKEN"
vault_role_env_var = "MLRUN_VAULT_ROLE"
token = None


class VaultStore:
    def __init__(self):
        self._token = None
        self.url = mlconf.vault.url or os.environ.get(vault_url_env_var)

    def _login(self):
        if self._token:
            return

        self._token = os.environ.get(vault_token_env_var)
        if self._token:
            return

        vault_role = None
        role_env = os.environ.get(vault_role_env_var)
        if role_env:
            role_type, role_val = role_env.split(":", 1)
            vault_role = "mlrun-role-{}-{}".format(role_type, role_val)

        self._login_with_jwt_token(vault_role)
        if self._token is None:
            logger.warning(
                "Vault login: no vault token is available. No secrets will be accessible"
            )

    @staticmethod
    def _generate_vault_path(
        prefix=vault_default_prefix,
        user=None,
        project=None,
        user_prefix="users",
        project_prefix="projects",
    ):
        full_path = prefix + "/mlrun/{}/{}"
        if user:
            return full_path.format(user_prefix, user)
        elif project:
            return full_path.format(project_prefix, project)
        else:
            raise ValueError(
                "To generate a vault secret path, either user or project must be specified"
            )

    @staticmethod
    def _read_jwt_token():
        # if for some reason the path to the token is not in conf, then attempt to get the SA token (works on k8s pods)
        token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
        if mlconf.vault.token_path:
            # Override the default SA token in case a specific token is installed in the mlconf-specified path
            secret_token_path = expanduser(mlconf.vault.token_path + "/token")
            if os.path.isfile(secret_token_path):
                logger.info(f"Using vault JWT token from {secret_token_path}")
                token_path = secret_token_path

        with open(token_path, "r") as token_file:
            jwt_token = token_file.read()

        return jwt_token

    def _vault_api_call(self, method, url, data=None):
        self._login()

        headers = {"X-Vault-Token": self._token}
        full_url = self.url + "/" + url
        if data:
            data = json.dumps(data)

        response = requests.request(method, full_url, headers=headers, data=data)

        if not response:
            logger.error(
                "Vault failed the API call. Response code: ({}) - {}".format(
                    response.status_code, response.reason
                )
            )
        return response

    # This method logins to the vault, assuming the container has a JWT token mounted as part of its assigned service
    # account.
    def _login_with_jwt_token(self, role):

        if role is None:
            logger.warning(
                "login_with_token: Role passed is None. Will not attempt login"
            )
            return

        jwt_token = self._read_jwt_token()

        login_url = f"{self.url}/v1/auth/kubernetes/login"
        data = {"jwt": jwt_token, "role": role}

        response = requests.post(login_url, data=json.dumps(data))
        if not response:
            logger.error(
                "login_with_token: Vault failed the login request. Role: {}, Response code: ({}) - {}".format(
                    role, response.status_code, response.reason
                )
            )
            return
        self._token = response.json()["auth"]["client_token"]

    def get_secrets(self, keys, user=None, project=None):
        secret_path = VaultStore._generate_vault_path(user=user, project=project)
        secrets = {}
        response = self._vault_api_call("GET", secret_path)

        if not response:
            return secrets

        values = response.json()["data"]["data"]

        # if no specific keys were asked for, return all the values available
        if not keys:
            return values

        for key in keys:
            if key in values:
                secrets[key] = values[key]
        return secrets

    def add_vault_secrets(self, items, project=None, user=None):
        data_object = {"data": items}
        url = VaultStore._generate_vault_path(project=project, user=user)

        response = self._vault_api_call("POST", url, data_object)
        if not response:
            raise ValueError(
                f"Vault failed the API call to create secrets. project={project}/user={user}"
            )

    def delete_vault_secrets(self, project=None, user=None):
        self._login()
        # Using the API to delete all versions + metadata of the given secret.
        url = "v1/secret/metadata/" + VaultStore._generate_vault_path(
            prefix="", project=project, user=user
        )

        response = self._vault_api_call("DELETE", url)
        if not response:
            raise ValueError(
                f"Vault failed the API call to delete secrets. project={project}/user={user}"
            )

    def create_project_policy(self, project):
        policy_name = "mlrun-project-{}".format(project)
        # TODO - need to make sure name is escaped properly and invalid chars are stripped
        url = "v1/sys/policies/acl/" + policy_name

        policy_str = (
            'path "secret/data/mlrun/projects/{0}" {{\n'
            + '  capabilities = ["read", "list", "create", "delete", "update"]\n'
            + "}}\n"
            + 'path "secret/data/mlrun/projects/{0}/*" {{\n'
            + '  capabilities = ["read", "list", "create", "delete", "update"]\n'
            + "}}"
        ).format(project)

        data_object = {"policy": policy_str}

        response = self._vault_api_call("PUT", url, data_object)
        if not response:
            raise ValueError(
                "Vault failed the API call to create a policy. Response code: ({}) - {}".format(
                    response.status_code, response.reason
                )
            )
        return policy_name

    def create_project_role(self, project, sa, policy, namespace="default-tenant"):
        role_name = "mlrun-role-project-{}".format(project)
        # TODO - need to make sure name is escaped properly and invalid chars are stripped
        url = "v1/auth/kubernetes/role/" + role_name

        role_object = {
            "bound_service_account_names": sa,
            "bound_service_account_namespaces": namespace,
            "policies": [policy],
            "token_ttl": 1800000,
        }

        response = self._vault_api_call("POST", url, role_object)
        if not response:
            raise ValueError(
                "Vault failed the API call to create a secret. Response code: ({}) - {}".format(
                    response.status_code, response.reason
                )
            )
        return role_name


def add_vault_project_secrets(project, items):
    return VaultStore().add_vault_secrets(items, project=project)


def add_vault_user_secrets(user, items):
    return VaultStore().add_vault_secrets(items, user=user)


def init_project_vault_configuration(project):
    """Create needed configurations for this new project:
    - Create a k8s service account with the name sa_vault_{proj name}
    - Create a Vault policy with the name proj_{proj name}
    - Create a Vault k8s auth role with the name role_proj_{proj name}
    These constructs will enable any pod created as part of this project to access the project's secrets
    in Vault, assuming that the secret which is part of the SA created is mounted to the pod.

    :param project: Project name
    """
    logger.info(f"init_project_vault_configuration called, project name: {project}")

    namespace = mlconf.namespace
    k8s = get_k8s_helper(silent=True)
    service_account_name = mlconf.vault.project_sa_name.format(project=project)

    secret_name = k8s.get_project_vault_secret_name(
        project, service_account_name, namespace=namespace
    )

    if not secret_name:
        k8s.create_project_service_account(service_account_name, namespace=namespace)

    vault = VaultStore()
    policy_name = vault.create_project_policy(project)
    role_name = vault.create_project_role(
        project, namespace=namespace, sa=service_account_name, policy=policy_name
    )

    logger.info("Created Vault policy: {}, role: {}".format(policy_name, role_name))
