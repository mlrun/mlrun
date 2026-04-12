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

import json

import deepdiff
import pytest

import mlrun.common.schemas
import mlrun.runtimes


def test_enum_yaml_dump():
    function = mlrun.new_function("function-name", kind="job")
    function.status.state = mlrun.common.schemas.FunctionState.ready
    print(function.to_yaml())


@pytest.mark.parametrize(
    "exclude_params,expected_result,is_empty",
    [
        (
            True,
            (
                '{"spec": {"notifications": [{"kind": "webhook", "name": "notification-test", "message": "completed", '
                '"severity": "info", "when": ["completed", "error"], "condition": ""}]}, "metadata": '
                '{"iteration": 0}, "status": {"state": "created"}}'
            ),
            False,
        ),
        (
            False,
            (
                '{"spec": {"notifications": [{"kind": "webhook", "name": "notification-test", "message": "completed", '
                '"severity": "info", "when": ["completed", "error"], "condition": "", "params": {"url": '
                '"https://url", "method": "PUT", "override_body": "AAAAAAAAAAAAAAAAAAAA"}}]}, '
                '"metadata": {"iteration": 0}, "status": {"state": "created"}}'
            ),
            False,
        ),
        (
            True,
            ('{"metadata": {"iteration": 0}, "status": {"state": "created"}}'),
            True,
        ),
        (
            False,
            ('{"metadata": {"iteration": 0}, "status": {"state": "created"}}'),
            True,
        ),
    ],
)
def test_runobject_to_json_with_exclude_params(
    exclude_params, expected_result, is_empty
):
    run_object_to_test = mlrun.model.RunObject()
    notification = mlrun.model.Notification(
        kind="webhook",
        when=["completed", "error"],
        name="notification-test",
        message="completed",
        condition="",
        severity="info",
        params={"url": "https://url", "method": "PUT", "override_body": "A" * 20},
    )

    run_object_to_test.spec.notifications = [] if is_empty else [notification]

    # Call the to_json function with the exclude_notifications_params parameter
    json_result = run_object_to_test.to_json(
        exclude_notifications_params=exclude_params
    )

    # Check if the JSON result matches the expected result
    assert (
        deepdiff.DeepDiff(
            json.loads(json_result), json.loads(expected_result), ignore_order=True
        )
        == {}
    )

    # Ensure the 'params' attribute of the notification is set back to the object
    if not is_empty:
        for notification in run_object_to_test.spec.notifications:
            assert notification.params


def test_image_builder_with_requirements_overwrite():
    """Test that ImageBuilder.with_requirements correctly handles the overwrite flag.

    Regression test: the original code had an operator precedence bug where
    `self.requirements or [] if not overwrite else []` was parsed as
    `self.requirements or ([] if not overwrite else [])`, which always evaluates
    to `self.requirements or []` regardless of the overwrite flag. This meant
    that overwrite=True never cleared existing requirements.
    """
    builder = mlrun.model.ImageBuilder()

    # Set initial requirements
    builder.with_requirements(requirements=["pandas", "numpy"])
    assert sorted(builder.requirements) == ["numpy", "pandas"]

    # Append without overwrite (default)
    builder.with_requirements(requirements=["scikit-learn"])
    assert sorted(builder.requirements) == ["numpy", "pandas", "scikit-learn"]

    # Overwrite=True should replace, not append
    builder.with_requirements(requirements=["requests"], overwrite=True)
    assert builder.requirements == ["requests"]


def test_image_builder_with_requirements_overwrite_empty():
    """Test that overwrite=True with empty requirements clears the list."""
    builder = mlrun.model.ImageBuilder()
    builder.with_requirements(requirements=["pandas"])
    assert builder.requirements == ["pandas"]

    # Overwrite with empty list should clear
    builder.with_requirements(requirements=[], overwrite=True)
    assert builder.requirements == []
