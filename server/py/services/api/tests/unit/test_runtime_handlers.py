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

from unittest.mock import MagicMock

import mlrun.common.constants as mlrun_constants

from services.api.common.runtime_handlers import get_resource_labels


def _make_function(name="my-func", project="my-project", labels=None):
    function = MagicMock()
    function.kind = "job"
    function.metadata.name = name
    function.metadata.project = project
    function.metadata.tag = "latest"
    function.metadata.labels = labels or {}
    return function


def _make_run(
    uid="abc123",
    name="my-run",
    project="my-project",
    owner=None,
    retry=None,
):
    run = MagicMock()
    run.metadata.uid = uid
    run.metadata.name = name
    run.metadata.project = project
    run.metadata.labels = {}
    if owner is not None:
        run.metadata.labels[mlrun_constants.MLRunInternalLabels.owner] = owner
    if retry is not None:
        run.metadata.labels[mlrun_constants.MLRunInternalLabels.retry] = retry
    return run


def test_owner_with_single_at_sign():
    """owner with a single '@' should be split into username and domain"""
    function = _make_function()
    run = _make_run(owner="user@example.com")
    labels = get_resource_labels(function, run)
    assert labels[mlrun_constants.MLRunInternalLabels.mlrun_owner] == "user"
    assert labels[mlrun_constants.MLRunInternalLabels.owner_domain] == "example.com"


def test_owner_with_multiple_at_signs():
    """owner with multiple '@' should split only on the first '@'"""
    function = _make_function()
    run = _make_run(owner="user@dept@company.com")
    labels = get_resource_labels(function, run)
    assert labels[mlrun_constants.MLRunInternalLabels.mlrun_owner] == "user"
    assert (
        labels[mlrun_constants.MLRunInternalLabels.owner_domain] == "dept@company.com"
    )


def test_owner_without_at_sign():
    """owner without '@' should be set as-is without a domain label"""
    function = _make_function()
    run = _make_run(owner="admin")
    labels = get_resource_labels(function, run)
    assert labels[mlrun_constants.MLRunInternalLabels.mlrun_owner] == "admin"
    assert mlrun_constants.MLRunInternalLabels.owner_domain not in labels


def test_no_owner():
    """run without an owner label should not set any owner or domain labels"""
    function = _make_function()
    run = _make_run(owner=None)
    labels = get_resource_labels(function, run)
    assert mlrun_constants.MLRunInternalLabels.mlrun_owner not in labels
    assert mlrun_constants.MLRunInternalLabels.owner_domain not in labels
