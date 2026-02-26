# Copyright 2025 Iguazio
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

import pytest

import mlrun
import mlrun.common.constants


def test_serving_to_job_auto_rename():
    """Test that ServingRuntime.to_job() auto-appends -batch suffix."""
    serving_fn = mlrun.new_function(name="test-serving", kind="serving")

    # Auto-generated name (default behavior)
    job = serving_fn.to_job()

    expected_name = f"test-serving{mlrun.common.constants.RESERVED_BATCH_JOB_SUFFIX}"
    assert job.metadata.name == expected_name, (
        f"Auto-generated job name should be '{expected_name}', got '{job.metadata.name}'"
    )
    assert serving_fn.metadata.name == "test-serving", (
        "Original serving function name should remain unchanged"
    )


def test_serving_to_job_custom_func_name():
    """Test that ServingRuntime.to_job() accepts custom func_name parameter."""
    serving_fn = mlrun.new_function(name="test-serving", kind="serving")

    # Custom func_name
    job = serving_fn.to_job(func_name="my-custom-batch-job")

    assert job.metadata.name == "my-custom-batch-job", (
        f"Custom job name should be 'my-custom-batch-job', got '{job.metadata.name}'"
    )
    assert serving_fn.metadata.name == "test-serving", (
        "Original serving function name should remain unchanged"
    )


def test_serving_to_job_metadata_independence():
    """Test that job metadata is independent from serving function metadata."""
    serving_fn = mlrun.new_function(name="test-serving", kind="serving")
    serving_fn.metadata.project = "original-project"

    job = serving_fn.to_job()

    # Modify job metadata
    job.metadata.project = "modified-project"

    # Original should be unchanged
    assert serving_fn.metadata.project == "original-project", (
        "Modifying job metadata should not affect original function"
    )


def test_local_to_job_metadata_independence():
    """Test that job metadata is independent from local function metadata."""
    local_fn = mlrun.new_function(name="test-local", kind="local", command="script.py")
    local_fn.metadata.project = "original-project"

    job = local_fn.to_job()

    # Modify job metadata
    job.metadata.project = "modified-project"

    # Original should be unchanged (verifies to_dict/from_dict creates independent objects)
    assert local_fn.metadata.project == "original-project", (
        "Modifying job metadata should not affect original function"
    )


def test_local_to_job_with_image():
    """Test that LocalRuntime.to_job() accepts image parameter."""
    local_fn = mlrun.new_function(name="test-local", kind="local", command="script.py")

    # Convert with custom image
    job = local_fn.to_job(image="custom-image:latest")

    assert job.spec.image == "custom-image:latest", (
        f"Job should have custom image, got '{job.spec.image}'"
    )


def test_local_to_job_custom_func_name():
    """Test that LocalRuntime.to_job() accepts custom func_name parameter."""
    local_fn = mlrun.new_function(name="test-local", kind="local", command="script.py")

    # Custom func_name
    job = local_fn.to_job(func_name="my-custom-job")

    assert job.metadata.name == "my-custom-job", (
        f"Custom job name should be 'my-custom-job', got '{job.metadata.name}'"
    )
    assert local_fn.metadata.name == "test-local", (
        "Original local function name should remain unchanged"
    )


def test_local_to_job_sets_kind_to_job():
    """Test that LocalRuntime.to_job() correctly sets kind to 'job'.

    Regression test for bug where to_job() was returning kind='local' instead of 'job'.
    """
    local_fn = mlrun.new_function(name="test-local", kind="local", command="script.py")

    # Verify local function has kind='local'
    assert local_fn.kind == "local", (
        f"Local function should have kind='local', got '{local_fn.kind}'"
    )

    # Convert to job
    job = local_fn.to_job()

    # Verify job has kind='job' (not 'local')
    assert job.kind == "job", f"Job should have kind='job', got '{job.kind}'"
    # LocalRuntime keeps the same name (no batch suffix)
    assert job.metadata.name == "test-local", (
        f"Job should keep original name, got '{job.metadata.name}'"
    )


def test_serving_to_job_sets_kind_to_job():
    """Test that ServingRuntime.to_job() correctly sets kind to 'job'."""
    serving_fn = mlrun.new_function(name="test-serving", kind="serving")

    # Verify serving function has kind='serving'
    assert serving_fn.kind == "serving", (
        f"Serving function should have kind='serving', got '{serving_fn.kind}'"
    )

    # Convert to job
    job = serving_fn.to_job()

    # Verify job has kind='job' (not 'serving')
    assert job.kind == "job", f"Job should have kind='job', got '{job.kind}'"
    assert job.metadata.name == "test-serving-batch", (
        f"Job should have batch suffix, got '{job.metadata.name}'"
    )


def test_to_job_preserves_class_attributes():
    """Test that to_job() correctly handles all class attributes and doesn't lose data.

    This test verifies that the to_dict()/from_dict() approach doesn't cause issues with:
    - Class attributes that should change (like `kind`)
    - Class attributes that are not serialized (like `_is_remote`, `_is_nested`)
    - Spec type conversion (FunctionSpec -> KubeResourceSpec)
    """

    # Create local function
    local_fn = mlrun.new_function(name="test", kind="local", command="test.py")

    # Verify LocalRuntime attributes before conversion
    assert local_fn.kind == "local"
    assert not local_fn._is_remote
    assert not local_fn._is_nested
    assert type(local_fn.spec).__name__ == "FunctionSpec"

    # Convert to job
    job = local_fn.to_job()

    #  Verify kind changed correctly (this was the bug we fixed)
    assert job.kind == "job", (
        f"kind should change from 'local' to 'job', got '{job.kind}'"
    )

    #  Verify _is_remote changed correctly (class attribute from KubejobRuntime)
    assert job._is_remote, f"_is_remote should be True for job, got {job._is_remote}"

    #  Verify _is_nested changed correctly (class attribute from KubejobRuntime)
    assert job._is_nested, f"_is_nested should be True for job, got {job._is_nested}"

    #  Verify spec type changed correctly (FunctionSpec -> KubeResourceSpec)
    assert type(job.spec).__name__ == "KubeResourceSpec", (
        f"spec should be KubeResourceSpec, got {type(job.spec).__name__}"
    )

    #  Verify spec has KubeResource-specific attributes
    assert hasattr(job.spec, "volumes"), (
        "KubeResourceSpec should have 'volumes' attribute"
    )
    assert hasattr(job.spec, "volume_mounts"), (
        "KubeResourceSpec should have 'volume_mounts' attribute"
    )
    assert hasattr(job.spec, "node_selector"), (
        "KubeResourceSpec should have 'node_selector' attribute"
    )

    #  Verify metadata is properly copied (not shared reference)
    local_fn.metadata.labels["test"] = "original"
    assert "test" not in job.metadata.labels, (
        "Metadata should be independent (not shared reference)"
    )

    #  Verify spec is properly copied (not shared reference)
    original_command = job.spec.command
    local_fn.spec.command = "modified.py"
    assert job.spec.command == original_command, (
        "Spec should be independent (not shared reference)"
    )

    print(" All class attributes and object independence verified!")


def test_to_job_name_length_validation_fails():
    """Test that to_job() fails when name+suffix exceeds 63 chars for serving."""
    # Create a function with name that will exceed limit when suffix is added
    # 58 chars + 6 chars ("-batch") = 64 chars > 63 limit
    long_name = "a" * 58
    fn = mlrun.new_function(name=long_name, kind="serving")

    # Should fail with clear error message
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match="Cannot convert serving function .* to batch job",
    ) as exc_info:
        fn.to_job()

    error_msg = str(exc_info.value)
    assert "exceeds Kubernetes limit" in error_msg
    assert "func_name parameter" in error_msg
    assert "63 characters" in error_msg


def test_to_job_name_length_validation_with_custom_name():
    """Test that providing func_name bypasses length validation for serving."""
    # Create a function with name that would exceed limit
    long_name = "a" * 58
    fn = mlrun.new_function(name=long_name, kind="serving")

    # Should succeed with custom name that fits within limit
    job = fn.to_job(func_name="custom-job-name")
    assert job.metadata.name == "custom-job-name"


def test_to_job_backward_compatibility():
    """Test backward compatibility - serving functions can be created with 58-63 char names.

    This ensures we didn't break existing code that creates serving functions
    with long names. They only fail when to_job() is called and suffix would exceed limit.
    """
    # Serving functions can be created with names up to 63 chars (standard K8s limit)
    long_name = "a" * 63
    serving_fn = mlrun.new_function(name=long_name, kind="serving")
    assert serving_fn.metadata.name == long_name

    # 58 char name also succeeds at creation (but would fail at to_job())
    name_58 = "a" * 58
    serving_fn = mlrun.new_function(name=name_58, kind="serving")
    assert serving_fn.metadata.name == name_58

    # 57 char name succeeds at both creation AND to_job()
    name_57 = "a" * 57
    serving_fn = mlrun.new_function(name=name_57, kind="serving")
    job = serving_fn.to_job()
    assert len(job.metadata.name) == 63  # 57 + 6 ("-batch")

    # Local functions don't have batch suffix logic, so they keep the same name
    local_fn = mlrun.new_function(name=long_name, kind="local")
    assert local_fn.metadata.name == long_name
    job = local_fn.to_job()
    assert job.metadata.name == long_name  # Name unchanged
