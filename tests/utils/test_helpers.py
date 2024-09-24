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
import asyncio
import re
import unittest.mock
from contextlib import nullcontext as does_not_raise

import pytest
from pandas import Timedelta, Timestamp

import mlrun.errors
import mlrun.utils.regex
import mlrun.utils.version
from mlrun.config import config
from mlrun.datastore.store_resources import parse_store_uri
from mlrun.utils import logger
from mlrun.utils.helpers import (
    StorePrefix,
    enrich_image_url,
    extend_hub_uri_if_needed,
    get_parsed_docker_registry,
    get_pretty_types_names,
    get_regex_list_as_string,
    resolve_image_tag_suffix,
    str_to_timestamp,
    template_artifact_path,
    update_in,
    validate_artifact_key_name,
    validate_tag_name,
    validate_v3io_stream_consumer_group,
    verify_field_regex,
    verify_list_items_type,
)

STORE_PREFIX = "store://{kind}/dummy-project/dummy-db-key"
ARTIFACT_STORE_PREFIX = STORE_PREFIX.format(kind=StorePrefix.Artifact)
DATASET_STORE_PREFIX = STORE_PREFIX.format(kind=StorePrefix.Dataset)
MODEL_STORE_PREFIX = STORE_PREFIX.format(kind=StorePrefix.Model)


def test_retry_until_successful_fatal_failure():
    original_exception = Exception("original")

    def _raise_fatal_failure():
        raise mlrun.errors.MLRunFatalFailureError(original_exception=original_exception)

    with pytest.raises(Exception, match=str(original_exception)):
        mlrun.utils.helpers.retry_until_successful(
            0, 1, logger, True, _raise_fatal_failure
        )


def test_retry_until_successful_sync():
    counter = 0

    def increase_counter():
        nonlocal counter
        counter += 1
        if counter < 3:
            raise Exception("error")

    mlrun.utils.helpers.retry_until_successful(0, 3, logger, True, increase_counter)


@pytest.mark.asyncio
async def test_retry_until_successful_async():
    counter = 0

    async def increase_counter():
        await asyncio.sleep(0.1)
        nonlocal counter
        counter += 1
        if counter < 3:
            raise Exception("error")

    await mlrun.utils.helpers.retry_until_successful_async(
        0, 3, logger, True, increase_counter
    )


@pytest.mark.parametrize(
    "value, expected",
    [
        ("asd", does_not_raise()),
        ("Asd", does_not_raise()),
        ("AsA", does_not_raise()),
        ("As-123_2.8A", does_not_raise()),
        ("1As-123_2.8A5", does_not_raise()),
        (
            "azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azs",
            does_not_raise(),
        ),
        (
            # Invalid because the first letter is -
            "-As-123_2.8A",
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        (
            # Invalid because the last letter is .
            "As-123_2.8A.",
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        (
            # Invalid because $ is not allowed
            "As-123_2.8A$a",
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        (
            # Invalid because it's more than 63 characters
            "azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsx",
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
    ],
)
def test_run_name_regex(value, expected):
    with expected:
        verify_field_regex("test_field", value, mlrun.utils.regex.run_name)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("{{pipelineparam:op=;name=mem}}", does_not_raise()),
        ("{{pipelineparam:op=2;name=mem}}", does_not_raise()),
        ("{{pipelineparam:op=10Mb;name=mem}}", does_not_raise()),
    ],
)
def test_pipeline_param(value, expected):
    with expected:
        verify_field_regex("test_field", value, mlrun.utils.regex.pipeline_param)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("asd", does_not_raise()),
        ("asdlnasd-123123-asd", does_not_raise()),
        # DNS-1035
        ("t012312-asdasd", does_not_raise()),
        (
            # Starts with alphanumeric number
            "012312-asdasd",
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        ("As-123_2.8A", pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
        ("1As-123_2.8A5", pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
        (
            # Invalid because the first letter is -
            "-As-123_2.8A",
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        (
            # Invalid because the last letter is .
            "As-123_2.8A.",
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        (
            # Invalid because $ is not allowed
            "As-123_2.8A$a",
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        # sprakjob length 29
        ("asdnoinasoidas-asdaskdlnaskdl", does_not_raise()),
        (
            "asdnoinasoidas-asdaskdlnaskdl2",
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
    ],
)
def test_spark_job_name_regex(value, expected):
    with expected:
        verify_field_regex("test_field", value, mlrun.utils.regex.sparkjob_name)


@pytest.mark.parametrize(
    "case",
    [
        {
            "input_uri": "http://no-hub-prefix",
            "expected_output": "http://no-hub-prefix",
        },
        {
            "input_uri": "hub://function_name",
            "expected_output": "function_name/latest/src/function.yaml",
        },
        {
            "input_uri": "hub://function_name:1.2.3",
            "expected_output": "function_name/1.2.3/src/function.yaml",
        },
        {
            "input_uri": "hub://default/function-name",
            "expected_output": "function_name/latest/src/function.yaml",
        },
        {
            "input_uri": "hub://default/function-name:3.4.5",
            "expected_output": "function_name/3.4.5/src/function.yaml",
        },
    ],
)
def test_extend_hub_uri(rundb_mock, case):
    hub_url = mlrun.mlconf.get_default_hub_source()
    input_uri = case["input_uri"]
    expected_output = case["expected_output"]
    output, is_hub_url = extend_hub_uri_if_needed(input_uri)
    if is_hub_url:
        expected_output = hub_url + expected_output
    assert expected_output == output


@pytest.mark.parametrize(
    "regex_list,value,expected_str,expected",
    [
        (
            [r"^.{0,9}$", r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$"],
            "blabla123",
            "(?=^.{0,9}$)(?=^[a-z0-9]([-a-z0-9]*[a-z0-9])?$).*$",
            True,
        ),
        (
            [r"^.{0,6}$", r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$"],
            "blabla123",
            "(?=^.{0,6}$)(?=^[a-z0-9]([-a-z0-9]*[a-z0-9])?$).*$",
            False,
        ),
        (
            [r"^.{0,6}$", r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$"],
            "bla^%",
            "(?=^.{0,6}$)(?=^[a-z0-9]([-a-z0-9]*[a-z0-9])?$).*$",
            False,
        ),
        (
            [r"^.{0,6}$", r"^a...e$", r"ab*"],
            "abcde",
            "(?=^.{0,6}$)(?=^a...e$)(?=ab*).*$",
            True,
        ),
        (
            [r"^.{0,6}$", r"^a...e$", r"ab*"],
            "abababe",
            "(?=^.{0,6}$)(?=^a...e$)(?=ab*).*$",
            False,
        ),
        (
            [r"^.{0,6}$", r"^a...e$", r"ab*"],
            "bcea",
            "(?=^.{0,6}$)(?=^a...e$)(?=ab*).*$",
            False,
        ),
    ],
)
def test_get_regex_list_as_string(regex_list, value, expected_str, expected):
    regex_str = get_regex_list_as_string(regex_list)
    assert expected_str == regex_str
    match = re.match(regex_str, value)
    assert match if expected else match is None


@pytest.mark.parametrize(
    "tag_name,expected",
    [
        (
            "tag_with_char!@#",
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        (
            "tag^name",
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        (
            "(tagname)",
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        (
            "tagname%",
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        ("tagname2.0", does_not_raise()),
        ("tag-name", does_not_raise()),
        ("tag-NAME", does_not_raise()),
        ("tag_name", does_not_raise()),
    ],
)
def test_validate_tag_name(tag_name, expected):
    with expected:
        validate_tag_name(
            tag_name,
            field_name="artifact.metadata,tag",
        )


@pytest.mark.parametrize(
    "artifact_name,expected",
    [
        # Invalid names
        (
            "artifact/name",
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        (
            "/artifact-name",
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        (
            "artifact-name/",
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        (
            "artifact-name\\test",
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
        ("", pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
        # Valid names
        ("artifact-name2.0", does_not_raise()),
        ("artifact-name", does_not_raise()),
        ("artifact-name", does_not_raise()),
        ("artifact-name_chars@#$", does_not_raise()),
        ("artifactNAME", does_not_raise()),
    ],
)
def test_validate_artifact_name(artifact_name, expected):
    with expected:
        validate_artifact_key_name(
            artifact_name,
            field_name="artifact.key",
        )


@pytest.mark.parametrize(
    "value,expected",
    [
        ("a", does_not_raise()),
        ("a_b", does_not_raise()),
        ("_a_b", pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
    ],
)
def test_validate_v3io_consumer_group(value, expected):
    with expected:
        validate_v3io_stream_consumer_group(
            value,
        )


@pytest.mark.parametrize(
    "case",
    [
        {
            "image": "mlrun/mlrun",
            "expected_output": "ghcr.io/mlrun/mlrun:0.5.2-unstable-adsf76s",
        },
        {
            "image": "mlrun/mlrun:some_tag",
            "expected_output": "ghcr.io/mlrun/mlrun:some_tag",
        },
        {
            "image": "quay.io/mlrun/mlrun",
            "expected_output": "quay.io/mlrun/mlrun:0.5.2-unstable-adsf76s",
        },
        {
            "image": "quay.io/mlrun/mlrun:some_tag",
            "expected_output": "quay.io/mlrun/mlrun:some_tag",
        },
        {
            "image": "mlrun/ml-models",
            "expected_output": "ghcr.io/mlrun/ml-models:0.5.2-unstable-adsf76s",
        },
        {
            "image": "mlrun/ml-models:some_tag",
            "expected_output": "ghcr.io/mlrun/ml-models:some_tag",
        },
        {
            "image": "quay.io/mlrun/ml-models",
            "expected_output": "quay.io/mlrun/ml-models:0.5.2-unstable-adsf76s",
        },
        {
            "image": "quay.io/mlrun/ml-models:some_tag",
            "expected_output": "quay.io/mlrun/ml-models:some_tag",
        },
        {"image": "fake_mlrun/ml-models", "expected_output": "fake_mlrun/ml-models"},
        {"image": "some_repo/some_image", "expected_output": "some_repo/some_image"},
        {
            "image": "python:3.9",
            "expected_output": "dummy-repo/python:3.9",
        },
        {
            "image": "some-repo/some-image",
            "expected_output": "dummy-repo/some-repo/some-image",
            "images_to_enrich_registry": "some-repo/some-image",
        },
        {
            "image": "some-repo/some-image:some-tag",
            "expected_output": "dummy-repo/some-repo/some-image:some-tag",
            "images_to_enrich_registry": "some-repo/some-image",
        },
        {
            "image": "mlrun/mlrun",
            "expected_output": "mlrun/mlrun:0.5.2-unstable-adsf76s",
            "images_to_enrich_registry": "some-repo/some-image",
        },
        {
            "image": "mlrun/mlrun",
            "expected_output": "ghcr.io/mlrun/mlrun:0.5.2-unstable-adsf76s",
            "images_to_enrich_registry": "some-repo/some-image,mlrun/mlrun",
        },
        {
            "image": "mlrun/mlrun:some-tag",
            "expected_output": "ghcr.io/mlrun/mlrun:some-tag",
            "images_to_enrich_registry": "some-repo/some-image,mlrun/mlrun",
        },
        {
            "image": "mlrun/ml-base",
            "expected_output": "ghcr.io/mlrun/ml-base:0.5.2-unstable-adsf76s",
            "images_to_enrich_registry": "mlrun/mlrun,mlrun/ml-base,mlrun/ml-models",
        },
        {
            "image": "mlrun/ml-base:0.5.2",
            "expected_output": "ghcr.io/mlrun/ml-base:0.5.2",
            "images_to_enrich_registry": "mlrun/mlrun:0.5.2,mlrun/ml-base:0.5.2,mlrun/ml-models:0.5.2",
        },
        {
            "image": "mlrun/ml-base",
            "expected_output": "ghcr.io/mlrun/ml-base:0.5.2-unstable-adsf76s",
            "images_to_enrich_registry": "^mlrun/mlrun:0.5.2-unstable-adsf76s,^mlrun/ml-base:0.5.2-unstable-adsf76s",
        },
        {
            "image": "quay.io/mlrun/ml-base",
            "expected_output": "quay.io/mlrun/ml-base:0.5.2-unstable-adsf76s",
            "images_to_enrich_registry": "^mlrun/mlrun:0.5.2-unstable-adsf76s,^mlrun/ml-base:0.5.2-unstable-adsf76s",
        },
        {
            "image": "mlrun/ml-base:0.5.2-unstable-adsf76s-another-tag-suffix",
            "expected_output": "ghcr.io/mlrun/ml-base:0.5.2-unstable-adsf76s-another-tag-suffix",
            "images_to_enrich_registry": "^mlrun/mlrun:0.5.2-unstable-adsf76s,^mlrun/ml-base:0.5.2-unstable-adsf76s",
        },
        {
            "image": "mlrun/ml-base:0.5.2-unstable-adsf76s-another-tag-suffix",
            "expected_output": "mlrun/ml-base:0.5.2-unstable-adsf76s-another-tag-suffix",
            "images_to_enrich_registry": "^mlrun/mlrun:0.5.2-unstable-adsf76s$,^mlrun/ml-base:0.5.2-unstable-adsf76s$",
        },
        {
            "image": "mlrun/mlrun",
            "expected_output": "mlrun/mlrun:0.5.2-unstable-adsf76s",
            "images_to_enrich_registry": "",
        },
        {
            "image": "mlrun/mlrun:bla",
            "expected_output": "ghcr.io/mlrun/mlrun:bla",
            "images_to_enrich_registry": "mlrun/mlrun",
            "images_registry": "ghcr.io",
        },
        {
            "image": "mlrun/mlrun:bla",
            "expected_output": "mlrun/mlrun:bla",
            "images_to_enrich_registry": "mlrun/mlrun",
            "images_registry": "",
        },
        {
            "image": "mlrun/mlrun:0.5.3",
            "expected_output": "mlrun/mlrun:0.5.3",
            "images_to_enrich_registry": "mlrun/mlrun:0.5.2",
        },
        {
            "image": "mlrun/mlrun",
            "expected_output": "ghcr.io/mlrun/mlrun:unstable",
            "images_tag": None,
            "version": "0.0.0+unstable",
        },
        {
            "image": "mlrun/mlrun",
            "expected_output": "ghcr.io/mlrun/mlrun:0.1.2-some-special-tag",
            "images_tag": None,
            "version": "0.1.2+some-special-tag",
        },
        {
            "image": "mlrun/mlrun",
            "client_version": "0.9.3-client-version",
            "images_tag": None,
            "expected_output": "mlrun/mlrun:0.9.3-client-version",
            "images_to_enrich_registry": "",
        },
        {
            "image": "mlrun/mlrun",
            "client_version": "0.9.3-client-version",
            "images_tag": "0.10.0-override-version",
            "expected_output": "mlrun/mlrun:0.10.0-override-version",
            "images_to_enrich_registry": "",
        },
        {
            "image": "mlrun/mlrun",
            "client_version": "0.9.3-client-version",
            "images_tag": "0.10.0-override-version",
            "version": "0.10.5-server-version",
            "expected_output": "mlrun/mlrun:0.10.0-override-version",
            "images_to_enrich_registry": "",
        },
        {
            "image": "mlrun/mlrun",
            "client_version": None,
            "images_tag": None,
            "version": "0.10.5-server-version",
            "expected_output": "mlrun/mlrun:0.10.5-server-version",
            "images_to_enrich_registry": "",
        },
        {
            "image": "mlrun/mlrun",
            "client_version": "0.9.3-client-version",
            "images_tag": None,
            "version": "0.10.5-server-version",
            "expected_output": "mlrun/mlrun:0.9.3-client-version",
            "images_to_enrich_registry": "",
        },
        {
            "image": "some/image",
            "client_version": "0.9.3-client-version",
            "images_tag": None,
            "version": "0.10.5-server-version",
            "expected_output": "some/image",
            "images_to_enrich_registry": "",
        },
        {
            "image": "mlrun/mlrun",
            "client_version": "1.3.0",
            "client_python_version": "3.9.0",
            "images_tag": None,
            "version": None,
            "expected_output": "mlrun/mlrun:1.3.0",
            "images_to_enrich_registry": "",
        },
        {
            "image": "mlrun/mlrun",
            "client_version": "1.3.0",
            "client_python_version": "3.7.13",
            "images_tag": None,
            "version": None,
            "expected_output": "mlrun/mlrun:1.3.0-py37",
            "images_to_enrich_registry": "",
        },
        {
            "image": "mlrun/mlrun",
            "client_version": "1.5.0",
            "client_python_version": "3.7.13",
            "images_tag": None,
            "version": None,
            "expected_output": "mlrun/mlrun:1.5.0",
            "images_to_enrich_registry": "",
        },
        {
            "image": "mlrun/mlrun",
            "client_version": "1.3.0",
            "client_python_version": None,
            "images_tag": None,
            "version": None,
            "expected_output": "mlrun/mlrun:1.3.0",
            "images_to_enrich_registry": "",
        },
        {
            "image": "mlrun/mlrun",
            "client_version": "1.3.0",
            "client_python_version": "3.9.13",
            "images_tag": None,
            "version": None,
            "expected_output": "mlrun/mlrun:1.3.0",
            "images_to_enrich_registry": "",
        },
        {
            "image": "mlrun/mlrun:1.2.0",
            "client_version": "1.3.0",
            "client_python_version": None,
            "images_tag": None,
            "version": None,
            "expected_output": "mlrun/mlrun:1.2.0",
            "images_to_enrich_registry": "",
        },
    ],
)
def test_enrich_image(case):
    default_images_to_enrich_registry = config.images_to_enrich_registry
    config.images_tag = case.get("images_tag", "0.5.2-unstable-adsf76s")
    config.images_registry = case.get("images_registry", "ghcr.io/")
    config.vendor_images_registry = case.get("vendor_images_registry", "dummy-repo/")
    config.images_to_enrich_registry = case.get(
        "images_to_enrich_registry", default_images_to_enrich_registry
    )
    if case.get("version") is not None:
        mlrun.utils.version.Version().get = unittest.mock.Mock(
            return_value={"version": case["version"]}
        )
    config.images_tag = case.get("images_tag", "0.5.2-unstable-adsf76s")
    image = case["image"]
    expected_output = case["expected_output"]
    client_version = case.get("client_version")
    client_python_version = case.get("client_python_version")
    output = enrich_image_url(image, client_version, client_python_version)
    assert output == expected_output


@pytest.mark.parametrize(
    "mlrun_version,python_version,expected",
    [
        ("1.3.0", "3.7.13", "-py37"),
        ("1.3.0", "3.9.13", ""),
        ("1.3.0", None, ""),
        ("1.3.0", "3.8.13", ""),
        ("1.3.0", "3.9.0", ""),
        ("1.2.0", "3.7.0", ""),
        ("1.2.0", "3.8.0", ""),
        ("1.3.0-rc12", "3.7.13", "-py37"),
        ("1.3.0-rc12", "3.9.13", ""),
        ("1.3.0-rc12", None, ""),
        ("1.3.0-rc12", "3.8.13", ""),
        ("1.3.1", "3.7.13", "-py37"),
        ("1.3.1", "3.9.13", ""),
        ("1.3.1", None, ""),
        ("1.3.1", "3.8.13", ""),
        ("1.3.1-rc12", "3.7.13", "-py37"),
        ("1.3.1-rc12", "3.9.13", ""),
        # an example of a version which contains a suffix of commit hash and not a rc suffix (our CI uses this format)
        ("1.3.0-zwqeiubz", "3.7.13", "-py37"),
        ("1.3.0-zwqeiubz", "3.9.13", ""),
        # an example of a dev version which contains `unstable` and not a rc suffix (When compiling from source without
        # defining a version)
        ("0.0.0-unstable", "3.7.13", "-py37"),
        ("0.0.0-unstable", "3.9.13", ""),
        # list of versions which are later than 1.3.0, if we decide to stop supporting python 3.7 in later versions
        # we can remove them
        ("1.4.0", "3.9.13", ""),
        ("1.4.0", "3.7.13", "-py37"),
        ("1.4.0-rc1", "3.7.13", "-py37"),
        ("1.4.0-rc1", "3.9.13", ""),
    ],
)
def test_resolve_image_tag_suffix(mlrun_version, python_version, expected):
    assert resolve_image_tag_suffix(mlrun_version, python_version) == expected


@pytest.mark.parametrize(
    "case",
    [
        {"docker_registry": "", "expected_registry": "", "expected_repository": None},
        {
            "docker_registry": "hedi/ingber",
            "expected_registry": None,
            "expected_repository": "hedi/ingber",
        },
        {
            "docker_registry": "localhost/hedingber",
            "expected_registry": "localhost",
            "expected_repository": "hedingber",
        },
        {
            "docker_registry": "gcr.io/hedingber",
            "expected_registry": "gcr.io",
            "expected_repository": "hedingber",
        },
        {
            "docker_registry": "local-registry:80/hedingber",
            "expected_registry": "local-registry:80",
            "expected_repository": "hedingber",
        },
        {
            "docker_registry": "docker-registry.default-tenant.app.hedingber-30-1.iguazio-cd1.com:80/hedingber",
            "expected_registry": "docker-registry.default-tenant.app.hedingber-30-1.iguazio-cd1.com:80",
            "expected_repository": "hedingber",
        },
        {
            "docker_registry": "docker-registry.default-tenant.app.hedingber-30-1.iguazio-cd1.com:80",
            "expected_registry": "docker-registry.default-tenant.app.hedingber-30-1.iguazio-cd1.com:80",
            "expected_repository": None,
        },
        {
            "docker_registry": "quay.io/",
            "expected_registry": "quay.io",
            "expected_repository": "",
        },
    ],
)
def test_get_parsed_docker_registry(case):
    config.httpdb.builder.docker_registry = case["docker_registry"]
    registry, repository = get_parsed_docker_registry()
    assert case["expected_registry"] == registry
    assert case["expected_repository"] == repository


@pytest.mark.parametrize(
    "uri,expected_output",
    [
        ("store:///123", (StorePrefix.Artifact, "123")),
        ("store://xyz", (StorePrefix.Artifact, "xyz")),
        (
            "store://feature-sets/123",
            (StorePrefix.FeatureSet, "123"),
        ),
        (
            "store://feature-vectors/456",
            (StorePrefix.FeatureVector, "456"),
        ),
        (
            "store://artifacts/890",
            (StorePrefix.Artifact, "890"),
        ),
        ("xxx://xyz", (None, "")),
    ],
)
def test_parse_store_uri(uri, expected_output):
    output = parse_store_uri(uri)
    assert expected_output == output


@pytest.mark.parametrize(
    "case",
    [
        {
            "artifact_path": "v3io://just/regular/path",
            "expected_artifact_path": "v3io://just/regular/path",
        },
        {
            "artifact_path": "v3io://path-with-unrealted-template/{{run.uid}}",
            "expected_artifact_path": "v3io://path-with-unrealted-template/project",
        },
        {
            "artifact_path": "v3io://template-project-not-provided/{{project}}",
            "raise": True,
        },
        {
            "artifact_path": "v3io://template-project-provided/{{project}}",
            "project": "some-project",
            "expected_artifact_path": "v3io://template-project-provided/some-project",
        },
        {
            "artifact_path": "v3io://legacy-template-project-provided/{{run.project}}",
            "project": "some-project",
            "expected_artifact_path": "v3io://legacy-template-project-provided/some-project",
        },
    ],
)
def test_template_artifact_path(case):
    if case.get("raise"):
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            template_artifact_path(case["artifact_path"], case.get("project"))
    else:
        filled_artifact_path = template_artifact_path(
            case["artifact_path"], case.get("project")
        )
        assert case["expected_artifact_path"] == filled_artifact_path


def test_update_in():
    obj = {}
    update_in(obj, "a.b.c", 2)
    assert obj["a"]["b"]["c"] == 2
    update_in(obj, "a.b.c", 3)
    assert obj["a"]["b"]["c"] == 3

    update_in(obj, "a.b.d", 3, append=True)
    assert obj["a"]["b"]["d"] == [3]
    update_in(obj, "a.b.d", 4, append=True)
    assert obj["a"]["b"]["d"] == [3, 4]


@pytest.mark.parametrize(
    "keys,val",
    [
        (
            ["meta", "label", "tags.data.com/env"],
            "value",
        ),
        (
            ["spec", "handler"],
            [1, 2, 3],
        ),
        (["metadata", "test", "labels", "test.data"], 1),
        (["metadata.test", "test.test", "labels", "test.data"], True),
        (["metadata", "test.middle.com", "labels", "test.data"], "data"),
    ],
)
def test_update_in_with_dotted_keys(keys, val):
    obj = {}
    # Join the keys list with dots to form a single key string.
    # If a key in the list has dots, wrap it with escaping (\\).
    key = ".".join([key if "." not in key else f"\\{key}\\" for key in keys])
    update_in(obj, key, val)
    for key in keys:
        obj = obj.get(key)
    assert obj == val


@pytest.mark.parametrize("actual_list", [[1], [1, "asd"], [None], ["asd", 23]])
@pytest.mark.parametrize("expected_types", [[str]])
def test_verify_list_types_failure(actual_list, expected_types):
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentTypeError):
        verify_list_items_type(actual_list, expected_types)


@pytest.mark.parametrize(
    "actual_list", [[1.0, 8, "test"], ["test", 0.0], [None], [[["test"], 23]]]
)
@pytest.mark.parametrize("expected_types", [[str, int]])
def test_verify_list_multiple_types_failure(actual_list, expected_types):
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentTypeError):
        verify_list_items_type(actual_list, expected_types)


@pytest.mark.parametrize("actual_list", [[], ["test"], ["test", "test1"]])
@pytest.mark.parametrize("expected_types", [[str]])
def test_verify_list_types_success(actual_list, expected_types):
    verify_list_items_type(actual_list, expected_types)


@pytest.mark.parametrize(
    "actual_list",
    [[1, 8, "test"], ["test", 0], [], ["test", 23, "test"], ["test"], [1], [123, 123]],
)
@pytest.mark.parametrize("expected_types", [[str, int]])
def test_verify_list_multiple_types_success(actual_list, expected_types):
    verify_list_items_type(actual_list, expected_types)


def test_get_pretty_types_names():
    cases = [
        ([], ""),
        ([str], "str"),
        ([str, int], "Union[str,int]"),
    ]
    for types, expected in cases:
        pretty_result = get_pretty_types_names(types)
        assert pretty_result == expected


def test_str_to_timestamp():
    now_time = Timestamp("2021-01-01 00:01:00")
    cases = [
        (None, None, None),
        ("1/1/2022", Timestamp("2022-01-01 00:00:00"), None),
        (Timestamp("1/1/2022"), Timestamp("1/1/2022"), None),
        ("not now", None, ValueError),
        (" now ", now_time, None),
        (" now floor 1H", now_time - Timedelta("1m"), None),
        ("now - 1d1h", now_time - Timedelta("1d1h"), None),
        ("now +1d1m", now_time + Timedelta("1d1m"), None),
        ("now +1d1m floor 1D", now_time + Timedelta("1d") - Timedelta("1m"), None),
        ("now * 1d1m", None, mlrun.errors.MLRunInvalidArgumentError),
        (
            "2022-01-11T18:28:00+00:00",
            Timestamp("2022-01-11 18:28:00+0000", tz="UTC"),
            None,
        ),
        (
            "2022-01-11T18:28:00-06:00",
            Timestamp("2022-01-11 18:28:00", tz="US/Central"),
            None,
        ),
    ]
    for time_str, expected, exception in cases:
        if exception is not None:
            with pytest.raises(exception):
                str_to_timestamp(time_str, now_time=now_time)
        else:
            timestamp = str_to_timestamp(time_str, now_time=now_time)
            print(time_str, timestamp, expected)
            assert timestamp == expected


def test_create_linear_backoff():
    stop_value = 120
    base = 2
    coefficient = 4
    backoff = mlrun.utils.helpers.create_linear_backoff(base, coefficient, stop_value)
    for i in range(0, 120):
        expected_value = min(base + i * coefficient, stop_value)
        assert expected_value, next(backoff)


def test_create_linear_backoff_negative_coefficient():
    stop_value = 2
    base = 120
    coefficient = -4
    backoff = mlrun.utils.helpers.create_linear_backoff(base, coefficient, stop_value)
    for i in range(120, 0):
        expected_value = min(base + i * coefficient, stop_value)
        assert expected_value, next(backoff)


def test_create_exponential_backoff():
    base = 2
    max_value = 120
    backoff = mlrun.utils.helpers.create_exponential_backoff(base, max_value)
    for i in range(1, 120):
        expected_value = min(base**i, max_value)
        assert expected_value, next(backoff)


def test_create_step_backoff():
    steps = [[2, 3], [10, 5], [120, None]]
    backoff = mlrun.utils.helpers.create_step_backoff(steps)
    for step in steps:
        step_value, step_occurrences = step
        if step_occurrences is not None:
            for _ in range(0, step_occurrences):
                assert step_value, next(backoff)
        else:
            # Run another 10 iterations:
            for _ in range(0, 10):
                assert step_value, next(backoff)


def test_retry_until_successful():
    def test_run(backoff):
        call_count = {"count": 0}
        unsuccessful_mock = unittest.mock.Mock()
        successful_mock = unittest.mock.Mock()

        def some_func(count_dict, a, b, some_other_thing=None):
            logger.debug(
                "Some function called", a=a, b=b, some_other_thing=some_other_thing
            )
            if count_dict["count"] < 3:
                logger.debug("Some function is still running, raising exception")
                count_dict["count"] += 1
                unsuccessful_mock()
                raise Exception("I'm running,try again later")

            logger.debug("Some function finished successfully")
            successful_mock()
            return "Finished"

        result = mlrun.utils.retry_until_successful(
            backoff,
            120,
            logger,
            True,
            some_func,
            call_count,
            5,
            [1, 8],
            some_other_thing="Just",
        )
        assert result, "Finished"
        assert unsuccessful_mock.call_count, 3
        assert successful_mock.call_count, 1

    test_run(0.02)

    test_run(mlrun.utils.create_linear_backoff(0.02, 0.02))


@pytest.mark.parametrize(
    "iterable_list, chunk_size, expected_chunked_list",
    [
        (["a", "b", "c"], 1, [["a"], ["b"], ["c"]]),
        (["a", "b", "c"], 2, [["a", "b"], ["c"]]),
        (["a", "b", "c"], 3, [["a", "b", "c"]]),
        (["a", "b", "c"], 4, [["a", "b", "c"]]),
        (["a", "b", "c"], 0, [["a", "b", "c"]]),
    ],
)
def test_iterate_list_by_chunks(iterable_list, chunk_size, expected_chunked_list):
    chunked_list = mlrun.utils.iterate_list_by_chunks(iterable_list, chunk_size)
    assert list(chunked_list) == expected_chunked_list


@pytest.mark.parametrize(
    "username,expected_normalized_username",
    [
        # sanity, all good
        ("test", "test"),
        # ensure ends with alphanumeric
        ("test.", "test"),
        ("test-", "test"),
        # lowercase
        ("TestUser", "testuser"),
        # remove special characters
        ("UserName!@#$", "username"),
        # dasherize
        ("user_name", "user-name"),
        ("User-Name_123", "user-name-123"),
        # everything with @ (email-like username)
        ("User_Name@domain.com", "user-name"),
        ("user@domain.com", "user"),
        ("user.name@example.com", "username"),
        ("user_name@example.com", "user-name"),
    ],
)
def test_normalize_username(username, expected_normalized_username):
    normalized_username = mlrun.utils.helpers.normalize_project_username(username)
    assert normalized_username == expected_normalized_username


@pytest.mark.parametrize(
    "basedir,path,is_symlink, is_valid",
    [
        ("/base", "/base/valid", False, True),
        ("/base", "/base/valid", True, True),
        ("/base", "/../invalid", True, False),
        ("/base", "/../invalid", False, False),
    ],
)
def test_is_safe_path(basedir, path, is_symlink, is_valid):
    safe = mlrun.utils.is_safe_path(basedir, path, is_symlink)
    assert safe == is_valid


@pytest.mark.parametrize(
    "kind, tag, target_path, expected",
    [
        (
            "artifact",
            "v1",
            "/path/to/artifact",
            f"{ARTIFACT_STORE_PREFIX}:v1@dummy-tree",
        ),
        (
            "artifact",
            None,
            "/path/to/artifact",
            f"{ARTIFACT_STORE_PREFIX}:latest@dummy-tree",
        ),
        (
            "artifact",
            "latest",
            "/path/to/artifact",
            f"{ARTIFACT_STORE_PREFIX}:latest@dummy-tree",
        ),
        ("dataset", "v1", "/path/to/artifact", f"{DATASET_STORE_PREFIX}:v1@dummy-tree"),
        (
            "dataset",
            None,
            "/path/to/artifact",
            f"{DATASET_STORE_PREFIX}:latest@dummy-tree",
        ),
        (
            "dataset",
            "latest",
            "/path/to/artifact",
            f"{DATASET_STORE_PREFIX}:latest@dummy-tree",
        ),
        ("model", "v1", "/path/to/artifact", f"{MODEL_STORE_PREFIX}:v1@dummy-tree"),
        ("model", None, "/path/to/artifact", f"{MODEL_STORE_PREFIX}:latest@dummy-tree"),
        (
            "model",
            "latest",
            "/path/to/artifact",
            f"{MODEL_STORE_PREFIX}:latest@dummy-tree",
        ),
        ("dir", "v1", "/path/to/artifact", "/path/to/artifact"),
        ("table", "v1", "/path/to/artifact", "/path/to/artifact"),
        ("plot", "v1", "/path/to/artifact", "/path/to/artifact"),
    ],
)
def test_get_artifact_target(kind, tag, target_path, expected):
    item = {
        "kind": kind,
        "spec": {
            "db_key": "dummy-db-key",
            "target_path": target_path,
        },
        "metadata": {"tree": "dummy-tree", "tag": tag},
    }
    target = mlrun.utils.get_artifact_target(item, project="dummy-project")
    assert target == expected


def test_validate_single_def_handler_invalid_handler():
    code = """
def handler():
    pass
def handler():
    pass
"""
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc:
        mlrun.utils.validate_single_def_handler("mlrun", code)
    assert str(exc.value) == (
        "The code file contains a function named “handler“, which is reserved. "
        + "Use a different name for your function."
    )


@pytest.mark.parametrize(
    "code",
    [
        """
def dummy_handler():
    pass
def handler():
    pass
""",
        """
def handler():
    pass
""",
        """
def handler():
    pass
def dummy_handler():
    def handler():
        pass
    handler()
""",
        """
# def handler():
#     pass
def handler():
    pass
""",
    ],
)
def test_validate_single_def_handler_valid_handler(code):
    try:
        mlrun.utils.validate_single_def_handler("mlrun", code)
    except mlrun.errors.MLRunInvalidArgumentError:
        pytest.fail(
            "validate_single_def_handler raised MLRunInvalidArgumentError unexpectedly."
        )
