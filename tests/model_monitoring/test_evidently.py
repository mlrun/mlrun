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

import warnings
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from uuid import uuid4

import pytest
import semver

from mlrun.errors import MLRunIncompatibleVersionError
from mlrun.model_monitoring.applications.evidently.base import (
    _check_evidently_version,
)

from .assets.custom_evidently_app import DemoEvidentlyMonitoringApp


@pytest.mark.parametrize(
    ("cur", "ref", "expectation"),
    [
        ("0.4.11", "0.4.11", does_not_raise()),
        ("0.4.12", "0.4.11", does_not_raise()),
        ("1.23.0", "1.1.32", does_not_raise()),
        ("0.4.11", "0.4.12", pytest.raises(MLRunIncompatibleVersionError)),
        ("0.4.11", "0.4.12", pytest.raises(MLRunIncompatibleVersionError)),
        ("1.0.3", "0.9.9", pytest.raises(MLRunIncompatibleVersionError)),
        ("0.6.0", "0.3.0", pytest.warns(UserWarning)),
        pytest.param("0.6.0", "0.3.0", does_not_raise(), marks=pytest.mark.xfail),
    ],
)
def test_version_check(
    cur: str,
    ref: str,
    expectation: AbstractContextManager,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with expectation:
            _check_evidently_version(
                cur=semver.Version.parse(cur), ref=semver.Version.parse(ref)
            )


def test_demo_evidently_app(tmpdir: Path) -> None:
    """Test that the workspace and the project's dashboards are created"""
    evidently_app = DemoEvidentlyMonitoringApp(
        evidently_project_id=uuid4(), evidently_workspace_path=str(tmpdir)
    )
    run = evidently_app.create_report_run(
        sample_df=evidently_app.train_set, schedule_time=None
    )
    added_run_uid = evidently_app.evidently_workspace.add_run(
        project_id=evidently_app.evidently_project_id, run=run
    )
    assert evidently_app.evidently_workspace.list_runs(
        project_id=evidently_app.evidently_project_id
    ) == [added_run_uid], "Different project runs than expected"
