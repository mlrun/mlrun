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

import base64
import json

import pytest

import mlrun
import mlrun.common.constants as mlrun_constants
import mlrun.runtimes.pod

from services.api.common.runtime_handlers import get_resource_labels


@pytest.mark.parametrize(
    "config_labels,function_labels,expected",
    [
        pytest.param({"team": "ml"}, {}, {"team": "ml"}, id="service-default"),
        pytest.param(
            {"team": "ml"}, {"team": "platform"}, {"team": "platform"}, id="user-wins"
        ),
        pytest.param(
            {"team": "ml"}, {"env": "dev"}, {"team": "ml", "env": "dev"}, id="merged"
        ),
        pytest.param({}, {"env": "dev"}, {"env": "dev"}, id="empty-config"),
    ],
)
def test_get_resource_labels_merges_service_defaults(
    monkeypatch, config_labels, function_labels, expected
):
    monkeypatch.setattr(
        mlrun.mlconf, "default_function_pod_labels", _b64(config_labels)
    )
    fn = _make_function(labels=function_labels)
    labels = get_resource_labels(fn)
    for key, value in expected.items():
        assert labels[key] == value
    assert labels[mlrun_constants.MLRunInternalLabels.project] == "my-proj"
    assert labels[mlrun_constants.MLRunInternalLabels.function] == "my-fn"


def test_get_resource_labels_system_labels_override_service_defaults(monkeypatch):
    # service defaults may include mlrun/* keys; the system layer overwrites
    # the ones it manages, while unrelated mlrun/* keys flow through.
    monkeypatch.setattr(
        mlrun.mlconf,
        "default_function_pod_labels",
        _b64(
            {
                f"{mlrun_constants.MLRUN_LABEL_PREFIX}project": "from-service-default",
                f"{mlrun_constants.MLRUN_LABEL_PREFIX}custom": "passthrough",
                "team": "ml",
            }
        ),
    )
    fn = _make_function(project="real-project")
    labels = get_resource_labels(fn)
    assert labels[mlrun_constants.MLRunInternalLabels.project] == "real-project"
    assert labels[f"{mlrun_constants.MLRUN_LABEL_PREFIX}custom"] == "passthrough"
    assert labels["team"] == "ml"


def _b64(d: dict) -> str:
    return base64.b64encode(json.dumps(d).encode()).decode()


def _make_function(
    name: str = "my-fn",
    project: str = "my-proj",
    tag: str = "v1",
    labels: dict | None = None,
):
    fn = mlrun.runtimes.pod.KubeResource()
    fn.metadata.name = name
    fn.metadata.project = project
    fn.metadata.tag = tag
    fn.metadata.labels = dict(labels or {})
    return fn
