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

import pytest

import mlrun
import mlrun.errors
import mlrun.runtimes.utils

_SPARK3_REGISTRY_IMAGE = "mckinsey-ig4-next-gen-docker-local.jfrog.io/spark-app"
_SPARK3_TAG = "3.5.6-scala2.12-java17-ubuntu-1"
_SPARK4_REGISTRY_IMAGE = "mckinsey-ig4-next-gen-docker-local.jfrog.io/spark-app"
_SPARK4_TAG = "4.2.0-scala2.13-java25-ubuntu-1"
_BUILT_FUNCTION_OPAQUE_IMAGE = ".sparkjob-from-github:latest"
_BUILT_FUNCTION_BASE_IMAGE = "iguazio/spark-app:3.5.5-b697"


@pytest.fixture
def spark_platform_image_config():
    original_image = mlrun.mlconf.spark_app_image
    original_tag = mlrun.mlconf.spark_app_image_tag
    yield
    mlrun.mlconf.spark_app_image = original_image
    mlrun.mlconf.spark_app_image_tag = original_tag


def _new_runtime() -> mlrun.runtimes.Spark3Runtime:
    runtime = mlrun.runtimes.Spark3Runtime()
    runtime.with_executor_requests(cpu=1, mem="512m")
    runtime.with_driver_requests(cpu=1, mem="512m")
    return runtime


def test_default_image_none_when_either_half_missing(spark_platform_image_config):
    runtime = _new_runtime()

    mlrun.mlconf.spark_app_image = ""
    mlrun.mlconf.spark_app_image_tag = ""
    assert runtime._default_image is None

    mlrun.mlconf.spark_app_image = _SPARK4_REGISTRY_IMAGE
    mlrun.mlconf.spark_app_image_tag = ""
    assert runtime._default_image is None

    mlrun.mlconf.spark_app_image = ""
    mlrun.mlconf.spark_app_image_tag = _SPARK4_TAG
    assert runtime._default_image is None


def test_default_image_combines_image_and_tag_when_both_set(
    spark_platform_image_config,
):
    runtime = _new_runtime()
    mlrun.mlconf.spark_app_image = _SPARK4_REGISTRY_IMAGE
    mlrun.mlconf.spark_app_image_tag = _SPARK4_TAG
    assert runtime._default_image == f"{_SPARK4_REGISTRY_IMAGE}:{_SPARK4_TAG}"


def test_default_image_appends_cuda_suffix_on_gpu_path(spark_platform_image_config):
    runtime = _new_runtime()
    runtime.with_executor_limits(cpu="2", gpus=1)
    mlrun.mlconf.spark_app_image = _SPARK4_REGISTRY_IMAGE
    mlrun.mlconf.spark_app_image_tag = _SPARK4_TAG
    assert runtime._default_image == f"{_SPARK4_REGISTRY_IMAGE}-cuda:{_SPARK4_TAG}"


@pytest.mark.parametrize(
    "registry_image, tag, expected_version",
    [
        (_SPARK3_REGISTRY_IMAGE, _SPARK3_TAG, "3.5.6"),
        (_SPARK4_REGISTRY_IMAGE, _SPARK4_TAG, "4.2.0"),
    ],
)
def test_default_image_and_effective_version_agree_for_spark3_and_spark4(
    spark_platform_image_config, registry_image, tag, expected_version
):
    runtime = _new_runtime()
    mlrun.mlconf.spark_app_image = registry_image
    mlrun.mlconf.spark_app_image_tag = tag

    default_image = runtime._default_image
    resolution = mlrun.runtimes.utils.resolve_spark_version(
        explicit_version=runtime.spec.spark_version,
        image=runtime.spec.image,
        base_image=runtime.spec.build.base_image,
    )

    assert (
        mlrun.runtimes.utils.extract_spark_version_from_image(default_image)
        == expected_version
    )
    assert resolution.effective_version == expected_version


def test_generated_default_image_derives_version_from_platform_tag(
    spark_platform_image_config,
):
    runtime = _new_runtime()
    runtime.spec.use_default_image = True
    mlrun.mlconf.spark_app_image = _SPARK4_REGISTRY_IMAGE
    mlrun.mlconf.spark_app_image_tag = _SPARK4_TAG

    assert runtime.default_mlrun_image == ".spark-job-default-image"

    resolution = mlrun.runtimes.utils.resolve_spark_version(
        explicit_version=runtime.spec.spark_version,
        image=runtime.spec.image,
        base_image=runtime.spec.build.base_image,
    )
    assert resolution.effective_version == "4.2.0"


def test_built_function_derives_version_from_base_image_over_opaque_final_image():
    runtime = _new_runtime()
    runtime.spec.image = _BUILT_FUNCTION_OPAQUE_IMAGE
    runtime.spec.build.base_image = _BUILT_FUNCTION_BASE_IMAGE

    resolution = mlrun.runtimes.utils.resolve_spark_version(
        explicit_version=runtime.spec.spark_version,
        image=runtime.spec.image,
        base_image=runtime.spec.build.base_image,
    )
    assert resolution.effective_version == "3.5.5"
    assert resolution.provenance_image == _BUILT_FUNCTION_BASE_IMAGE


def test_custom_opaque_image_with_no_version_source_fails():
    runtime = _new_runtime()
    runtime.spec.image = "mlrun/mlrun:latest"

    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mlrun.runtimes.utils.resolve_spark_version(
            explicit_version=runtime.spec.spark_version,
            image=runtime.spec.image,
            base_image=runtime.spec.build.base_image,
        )


def test_explicit_matching_image_version_pair_is_preserved():
    runtime = _new_runtime()
    runtime.spec.image = f"{_SPARK4_REGISTRY_IMAGE}:{_SPARK4_TAG}"
    runtime.spec.spark_version = "4.2.0"

    resolution = mlrun.runtimes.utils.resolve_spark_version(
        explicit_version=runtime.spec.spark_version,
        image=runtime.spec.image,
        base_image=runtime.spec.build.base_image,
    )
    assert resolution.effective_version == "4.2.0"


def test_explicit_version_cannot_reuse_platform_default_image_from_other_major():
    runtime = _new_runtime()
    runtime.spec.image = f"{_SPARK4_REGISTRY_IMAGE}:{_SPARK4_TAG}"
    runtime.spec.spark_version = "3.5.5"

    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mlrun.runtimes.utils.resolve_spark_version(
            explicit_version=runtime.spec.spark_version,
            image=runtime.spec.image,
            base_image=runtime.spec.build.base_image,
        )


def test_spark_version_survives_to_dict_and_from_dict_round_trip():
    runtime = _new_runtime()
    runtime.spec.spark_version = "4.2.0"

    round_tripped = mlrun.runtimes.Spark3Runtime.from_dict(runtime.to_dict())

    assert round_tripped.kind == "spark"
    assert round_tripped.spec.spark_version == "4.2.0"
    assert round_tripped.to_dict() == runtime.to_dict()
