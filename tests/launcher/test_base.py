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
import pytest

import mlrun.launcher.base


def test_validate_state_thresholds_success():
    mlrun.launcher.base.BaseLauncher._validate_state_thresholds(
        state_thresholds={
            "pending_scheduled": "-1",
            "running": "1000s",
            "image_pull_backoff": "3m",
        }
    )


def test_validate_state_thresholds_failure():
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc:
        mlrun.launcher.base.BaseLauncher._validate_state_thresholds(
            state_thresholds={
                "pending_scheduled": "-1",
                "running": "1000s",
                "image_pull_backoff": "3mm",
            }
        )
    assert (
        'Threshold for state image_pull_backoff must match the pattern ^(\\d+)([smhdw])$ or be "-1"'
        in str(exc.value)
    )

    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc:
        mlrun.launcher.base.BaseLauncher._validate_state_thresholds(
            state_thresholds={
                "pending_scheduled": -1,
            }
        )
    assert "Threshold for state pending_scheduled must be a string" in str(exc.value)
