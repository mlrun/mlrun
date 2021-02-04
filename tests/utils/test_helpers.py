from mlrun.config import config
from mlrun.datastore.store_resources import parse_store_uri
from mlrun.utils.helpers import (
    verify_field_regex,
    extend_hub_uri_if_needed,
    enrich_image_url,
    get_parsed_docker_registry,
    StorePrefix,
)
from mlrun.utils.regex import run_name


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
    ]
    config.images_registry = "ghcr.io/"
    config.images_tag = "0.5.2-unstable-adsf76s"
    for case in cases:
        image = case["image"]
        expected_output = case["expected_output"]
        output = enrich_image_url(image)
        assert expected_output == output


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
