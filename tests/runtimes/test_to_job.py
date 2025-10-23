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

import mlrun
import mlrun.common.constants


def test_serving_to_job_auto_rename():
    """Test that ServingRuntime.to_job() auto-appends -batch suffix."""
    serving_fn = mlrun.new_function(name="test-serving", kind="serving")

    # Auto-generated name (default behavior)
    job = serving_fn.to_job()

    expected_name = f"test-serving{mlrun.common.constants.RESERVED_BATCH_JOB_SUFFIX}"
    assert (
        job.metadata.name == expected_name
    ), f"Auto-generated job name should be '{expected_name}', got '{job.metadata.name}'"
    assert (
        serving_fn.metadata.name == "test-serving"
    ), "Original serving function name should remain unchanged"


def test_serving_to_job_custom_func_name():
    """Test that ServingRuntime.to_job() accepts custom func_name parameter."""
    serving_fn = mlrun.new_function(name="test-serving", kind="serving")

    # Custom func_name
    job = serving_fn.to_job(func_name="my-custom-batch-job")

    assert (
        job.metadata.name == "my-custom-batch-job"
    ), f"Custom job name should be 'my-custom-batch-job', got '{job.metadata.name}'"
    assert (
        serving_fn.metadata.name == "test-serving"
    ), "Original serving function name should remain unchanged"


def test_serving_to_job_custom_name_with_batch_suffix():
    """Test that custom func_name with -batch suffix is allowed."""
    serving_fn = mlrun.new_function(name="test-serving", kind="serving")

    # Custom func_name with -batch suffix (should be allowed for jobs)
    custom_name = f"another-job{mlrun.common.constants.RESERVED_BATCH_JOB_SUFFIX}"
    job = serving_fn.to_job(func_name=custom_name)

    assert (
        job.metadata.name == custom_name
    ), f"Custom job name with -batch suffix should be allowed, got '{job.metadata.name}'"


def test_serving_to_job_already_has_batch_suffix():
    """Test that to_job() doesn't double-append suffix if already present."""
    serving_fn = mlrun.new_function(name="test-serving", kind="serving")

    # First conversion
    job1 = serving_fn.to_job()
    expected_name = f"test-serving{mlrun.common.constants.RESERVED_BATCH_JOB_SUFFIX}"
    assert job1.metadata.name == expected_name

    # Convert the serving function again (not the job)
    job2 = serving_fn.to_job()
    assert job2.metadata.name == expected_name, "Should not double-append -batch suffix"


def test_serving_to_job_metadata_independence():
    """Test that job metadata is independent from serving function metadata."""
    serving_fn = mlrun.new_function(name="test-serving", kind="serving")
    serving_fn.metadata.project = "original-project"

    job = serving_fn.to_job()

    # Modify job metadata
    job.metadata.project = "modified-project"

    # Original should be unchanged
    assert (
        serving_fn.metadata.project == "original-project"
    ), "Modifying job metadata should not affect original function"


def test_local_to_job_auto_rename():
    """Test that LocalRuntime.to_job() auto-appends -batch suffix."""
    local_fn = mlrun.new_function(name="test-local", kind="local", command="script.py")

    # Auto-generated name
    job = local_fn.to_job()

    expected_name = f"test-local{mlrun.common.constants.RESERVED_BATCH_JOB_SUFFIX}"
    assert (
        job.metadata.name == expected_name
    ), f"Auto-generated job name should be '{expected_name}', got '{job.metadata.name}'"
    assert (
        local_fn.metadata.name == "test-local"
    ), "Original local function name should remain unchanged"


def test_local_to_job_custom_func_name():
    """Test that LocalRuntime.to_job() accepts custom func_name parameter."""
    local_fn = mlrun.new_function(name="test-local", kind="local", command="script.py")

    # Custom func_name
    job = local_fn.to_job(func_name="my-custom-job")

    assert (
        job.metadata.name == "my-custom-job"
    ), f"Custom job name should be 'my-custom-job', got '{job.metadata.name}'"


def test_local_to_job_metadata_independence():
    """Test that job metadata is independent from local function metadata."""
    local_fn = mlrun.new_function(name="test-local", kind="local", command="script.py")
    local_fn.metadata.project = "original-project"

    job = local_fn.to_job()

    # Modify job metadata
    job.metadata.project = "modified-project"

    # Original should be unchanged (verifies to_dict/from_dict creates independent objects)
    assert (
        local_fn.metadata.project == "original-project"
    ), "Modifying job metadata should not affect original function"


def test_local_to_job_with_image():
    """Test that LocalRuntime.to_job() accepts image parameter."""
    local_fn = mlrun.new_function(name="test-local", kind="local", command="script.py")

    # Convert with custom image
    job = local_fn.to_job(image="custom-image:latest")

    assert (
        job.spec.image == "custom-image:latest"
    ), f"Job should have custom image, got '{job.spec.image}'"
