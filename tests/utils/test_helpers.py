# Copyright 2018 Iguazio
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
    fill_artifact_path_template,
    get_parsed_docker_registry,
    get_pretty_types_names,
    get_regex_list_as_string,
    str_to_timestamp,
    validate_tag_name,
    verify_field_regex,
    verify_list_items_type,
)


def test_retry_until_successful_fatal_failure():
    original_exception = Exception("original")

    def _raise_fatal_failure():
        raise mlrun.errors.MLRunFatalFailureError(original_exception=original_exception)

    with pytest.raises(Exception, match=str(original_exception)):
        mlrun.utils.helpers.retry_until_successful(
            0, 1, logger, True, _raise_fatal_failure
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
            # Invalid because it's more then 63 characters
            "azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsx",
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
    ],
)
def test_run_name_regex(value, expected):
    with expected:
        verify_field_regex("test_field", value, mlrun.utils.regex.run_name)


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


def test_extend_hub_uri():
    hub_urls = [
        "https://raw.githubusercontent.com/mlrun/functions/{tag}/{name}/function.yaml",
        "https://raw.githubusercontent.com/mlrun/functions",
    ]
    cases = [
        {
            "input_uri": "http://no-hub-prefix",
            "expected_output": "http://no-hub-prefix",
        },
        {
            "input_uri": "hub://function_name",
            "expected_output": "https://raw.githubusercontent.com/mlrun/functions/master/function_name/function.yaml",
        },
        {
            "input_uri": "hub://function_name:development",
            "expected_output": "https://raw.githubusercontent.com/mlrun/functions/development/function_name/function.ya"
            "ml",
        },
        {
            "input_uri": "hub://function-name",
            "expected_output": "https://raw.githubusercontent.com/mlrun/functions/master/function_name/function.yaml",
        },
        {
            "input_uri": "hub://function-name:development",
            "expected_output": "https://raw.githubusercontent.com/mlrun/functions/development/function_name/function.ya"
            "ml",
        },
    ]
    for hub_url in hub_urls:
        mlrun.mlconf.hub_url = hub_url
        for case in cases:
            input_uri = case["input_uri"]
            expected_output = case["expected_output"]
            output, _ = extend_hub_uri_if_needed(input_uri)
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
            "tag_name",
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
        ),
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
    ],
)
def test_validate_tag_name(tag_name, expected):
    with expected:
        validate_tag_name(
            tag_name,
            field_name="artifact.metadata,tag",
        )


def test_enrich_image():
    cases = [
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
            "image": "some-repo/some-image",
            "expected_output": "ghcr.io/some-repo/some-image",
            "images_to_enrich_registry": "some-repo/some-image",
        },
        {
            "image": "some-repo/some-image:some-tag",
            "expected_output": "ghcr.io/some-repo/some-image:some-tag",
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
    ]
    default_images_to_enrich_registry = config.images_to_enrich_registry
    for case in cases:
        config.images_tag = case.get("images_tag", "0.5.2-unstable-adsf76s")
        config.images_registry = case.get("images_registry", "ghcr.io/")
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
        output = enrich_image_url(image, client_version)
        assert output == expected_output


def test_get_parsed_docker_registry():
    cases = [
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
    ]
    for case in cases:
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


def test_fill_artifact_path_template():
    cases = [
        {
            "artifact_path": "v3io://just/regular/path",
            "expected_artifact_path": "v3io://just/regular/path",
        },
        {
            "artifact_path": "v3io://path-with-unrealted-template/{{run.uid}}",
            "expected_artifact_path": "v3io://path-with-unrealted-template/{{run.uid}}",
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
    ]
    for case in cases:
        if case.get("raise"):
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                fill_artifact_path_template(case["artifact_path"], case.get("project"))
        else:
            filled_artifact_path = fill_artifact_path_template(
                case["artifact_path"], case.get("project")
            )
            assert case["expected_artifact_path"] == filled_artifact_path


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
