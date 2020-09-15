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
import json

vault_default_prefix = 'v1/secret/data'
vault_users_prefix = 'users'
vault_projects_prefix = 'projects'
vault_url_env_var = 'MLRUN_VAULT_URL'
vault_token_env_var = 'MLRUN_VAULT_TOKEN'


def get_vault_params():
    url = os.environ.get(vault_url_env_var) or 'https://vault.default-tenant.app.saarc-vault.iguazio-cd2.com'
    token = os.environ.get(vault_token_env_var)
    if token is None:
        raise OSError("error: no vault token available")
    return url, token


def generate_vault_path(prefix=vault_default_prefix,
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


def get_vault_secrets(keys,
                      user=None,
                      project=None):
    vault_url, token = get_vault_params()
    headers = {'X-Vault-Token': token}
    secret_path = generate_vault_path(user=user, project=project)

    query_url = vault_url + '/' + secret_path
    response = requests.get(query_url, headers=headers)
    if not response:
        raise ValueError("Vault failed the API call to retrieve secrets. Response code: ({}) - {}".
                         format(response.status_code, response.reason))
    values = response.json()['data']['data']
    resp = {}
    for key in keys:
        resp[key] = values[key]
    return resp


def add_vault_secret(items,
                     project=None,
                     user=None):
    vault_url, token = get_vault_params()
    headers = {'X-Vault-Token': token}

    data_obj = {"data": items}
    payload = json.dumps(data_obj)

    url = vault_url + '/' + generate_vault_path(project=project, user=user)
    response = requests.post(url, data=payload, headers=headers)
    if not response:
        raise ValueError("Vault failed the API call to create secrets. Response code: ({}) - {}".
                         format(response.status_code, response.reason))


def add_vault_project_secret(project, items):
    return add_vault_secret(items, project=project)


def add_vault_user_secret(user, items):
    return add_vault_secret(items, user=user)
