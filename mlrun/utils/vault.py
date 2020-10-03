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

vault_default_prefix = 'v1/secret/data'
vault_users_prefix = 'users'
vault_projects_prefix = 'projects'
vault_url_env_var = 'MLRUN_VAULT_URL'
vault_token_env_var = 'MLRUN_VAULT_TOKEN'
vault_role_env_var = 'MLRUN_VAULT_ROLE'
token = None


class VaultStore:
    def __init__(self):
        self._token = None
        self.url = mlconf.vault_url or os.environ.get(vault_url_env_var)

    def _login(self):
        if self._token:
            return

        self._token = os.environ.get(vault_token_env_var)
        if self._token:
            return

        vault_role = None
        role_env = os.environ.get(vault_role_env_var)
        if role_env:
            role_type, role_val = role_env.split(':', 1)
            vault_role = 'role_{}_{}'.format(role_type, role_val)

        self._login_with_token(vault_role)
        if self._token is None:
            logger.warning('warning: get_vault_params: no vault token is available. No secrets will be accessible')

    @staticmethod
    def _generate_vault_path(prefix=vault_default_prefix,
                             user=None,
                             project=None,
                             user_prefix=vault_users_prefix,
                             project_prefix=vault_projects_prefix):
        full_path = prefix + '/{}/{}'
        if user:
            return full_path.format(user_prefix, user)
        elif project:
            return full_path.format(project_prefix, project)
        else:
            raise ValueError("error: to generate a vault secret path, either user or project must be specified")

    def _read_jwt_token(self):
        # if for some reason the path to the token is not in conf, then attempt to get the SA token (works on k8s pods)
        token_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
        if mlconf.vault_token_path:
            # Override the default SA token in case a specific token is installed in the mlconf-specified path
            secret_token_path = expanduser(mlconf.vault_token_path + '/token')
            if os.path.isfile(secret_token_path):
                token_path = secret_token_path

        with open(token_path, "r") as token_file:
            jwt_token = token_file.read()

        return jwt_token

    # This method logins to the vault, assuming the container has a JWT token mounted as part of its assigned service
    # account.
    def _login_with_token(self,
                          role):

        if role is None:
            logger.warning('warning: login_with_token: Role passed is None. Will not attempt login')
            return

        jwt_token = self._read_jwt_token()

        login_url = f'{self.url}/v1/auth/kubernetes/login'
        data = {"jwt": jwt_token, "role": role}

        response = requests.post(login_url, data=json.dumps(data))
        if not response:
            logger.error('error: login_with_sa: Vault failed the login request. Response code: ({}) - {}'.
                         format(response.status_code, response.reason))
            return
        self._token = response.json()['auth']['client_token']

    def get_secrets(self,
                    keys,
                    user=None,
                    project=None):
        self._login()

        headers = {'X-Vault-Token': self._token}
        secret_path = VaultStore._generate_vault_path(user=user, project=project)

        resp = {}

        query_url = self.url + '/' + secret_path
        response = requests.get(query_url, headers=headers)
        if not response:
            logger.error('warning: Vault failed the API call to retrieve secrets. Response code: ({}) - {}'.
                         format(response.status_code, response.reason))
            return resp
        values = response.json()['data']['data']
        for key in keys:
            resp[key] = values[key]
        return resp

    def add_vault_secret(self,
                         items,
                         project=None,
                         user=None):
        self._login()
        headers = {'X-Vault-Token': self._token}

        data_obj = {"data": items}
        payload = json.dumps(data_obj)

        url = self.url + '/' + VaultStore._generate_vault_path(project=project, user=user)
        response = requests.post(url, data=payload, headers=headers)
        if not response:
            raise ValueError("Vault failed the API call to create secrets. Response code: ({}) - {}".
                             format(response.status_code, response.reason))

    def create_project_policy(self,
                              project):
        self._login()
        headers = {'X-Vault-Token': self._token}

        policy_name = 'proj_{}'.format(project)
        # TODO - need to make sure name is escaped properly and invalid chars are stripped
        url = self.url + "/v1/sys/policies/acl/" + policy_name

        policy_str = (
                '''path "secret/data/projects/{}" {{capabilities = ["read", "list", "create", "delete", "update"]}} ''' +
                '''path "secret/data/projects/{}/*" {{capabilities = ["read", "list", "create", "delete", "update"]}}'''
        ).format(project, project)

        data_obj = {"policy": policy_str}
        payload = json.dumps(data_obj)

        response = requests.put(url, data=payload, headers=headers)
        if not response:
            raise ValueError("Vault failed the API call to create a policy. Response code: ({}) - {}".
                             format(response.status_code, response.reason))
        return policy_name

    def create_project_role(self,
                            project, sa, policy,
                            namespace='default-tenant'):
        self._login()
        headers = {'X-Vault-Token': self._token}
        role_name = 'role_proj_{}'.format(project)
        # TODO - need to make sure name is escaped properly and invalid chars are stripped
        url = self.url + "/v1/auth/kubernetes/role/" + role_name

        role_obj = {"bound_service_account_names": sa,
                    "bound_service_account_namespaces": namespace,
                    "policies": [policy],
                    "token_ttl": 1800000
                    }
        payload = json.dumps(role_obj)

        response = requests.post(url, data=payload, headers=headers)
        if not response:
            raise ValueError("Vault failed the API call to create a secret. Response code: ({}) - {}".
                             format(response.status_code, response.reason))
        return role_name


def add_vault_project_secret(project, items):
    return VaultStore().add_vault_secret(items, project=project)


def add_vault_user_secret(user, items):
    return VaultStore().add_vault_secret(items, user=user)
