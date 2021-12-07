import unittest.mock

import pytest

import mlrun.errors
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
    verify_field_regex,
    verify_list_items_type,
)
from mlrun.utils.regex import run_name


def test_retry_until_successful_fatal_failure():
    original_exception = Exception("original")

    def _raise_fatal_failure():
        raise mlrun.utils.helpers.FatalFailureException(original_exception)

    with pytest.raises(Exception, match=str(original_exception)):
        mlrun.utils.helpers.retry_until_successful(
            0, 1, logger, True, _raise_fatal_failure
        )


def test_run_name_regex():
    cases = [
        {"value": "asd", "valid": True},
        {"value": "Asd", "valid": True},
        {"value": "AsA", "valid": True},
        {"value": "As-123_2.8A", "valid": True},
        {"value": "1As-123_2.8A5", "valid": True},
        {
            "value": "azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azs",
            "valid": True,
        },
        {
            # Invalid because the first letter is -
            "value": "-As-123_2.8A",
            "valid": False,
        },
        {
            # Invalid because the last letter is .
            "value": "As-123_2.8A.",
            "valid": False,
        },
        {
            # Invalid because $ is not allowed
            "value": "As-123_2.8A$a",
            "valid": False,
        },
        {
            # Invalid because it's more then 63 characters
            "value": "azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsx",
            "valid": False,
        },
    ]
    for case in cases:
        try:
            verify_field_regex("test_field", case["value"], run_name)
        except Exception:
            if case["valid"]:
                raise


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
        output = enrich_image_url(image)
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


def test_parse_store_uri():
    cases = [
        {"uri": "store:///123", "expected_output": (StorePrefix.Artifact, "123")},
        {"uri": "store://xyz", "expected_output": (StorePrefix.Artifact, "xyz")},
        {
            "uri": "store://feature-sets/123",
            "expected_output": (StorePrefix.FeatureSet, "123"),
        },
        {
            "uri": "store://feature-vectors/456",
            "expected_output": (StorePrefix.FeatureVector, "456"),
        },
        {
            "uri": "store://artifacts/890",
            "expected_output": (StorePrefix.Artifact, "890"),
        },
        {"uri": "xxx://xyz", "expected_output": (None, "")},
    ]
    for case in cases:
        output = parse_store_uri(case["uri"])
        assert case["expected_output"] == output


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
