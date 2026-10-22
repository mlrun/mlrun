# Copyright 2026 Iguazio
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

import copy

import pytest

import mlrun
import mlrun.common.schemas
import mlrun.config
import mlrun.projects.pipelines


@pytest.mark.parametrize(
    "synced_security_context, expected_env",
    [
        (
            {"non_root_home": "/home/custom"},
            [{"name": "HOME", "value": "/home/custom"}],
        ),
        ({"non_root_home": ""}, []),
        # older servers omit the key, so the SDK default applies
        ({}, [{"name": "HOME", "value": "/tmp"}]),
    ],
)
def test_enrich_kfp_pod_security_context_non_root_home_after_client_spec_sync(
    synced_security_context, expected_env
):
    # match the security context installed by HTTPRunDB.connect()
    mlrun.mlconf.function.spec.security_context = {
        "default": "e30=",
        "enrichment_mode": mlrun.common.schemas.SecurityContextEnrichmentModes.override.value,
        "enrichment_group_id": 65534,
        "pipelines": {"kfp_pod_user_unix_id": 5},
        **synced_security_context,
    }
    kfp_pod_template = {"container": {}}

    try:
        mlrun.projects.pipelines._enrich_kfp_pod_security_context(
            kfp_pod_template, mlrun.new_function("func", kind="job")
        )

        assert kfp_pod_template["SecurityContext"] == {
            "runAsUser": 5,
            "runAsGroup": 65534,
        }
        assert kfp_pod_template["container"]["env"] == expected_env
    finally:
        # reloading config does not restore keys omitted by the synced context
        mlrun.mlconf.function.spec.security_context = copy.deepcopy(
            mlrun.config.default_config["function"]["spec"]["security_context"]
        )
