import kubernetes.client as k8s_client
import pytest

import mlrun


def _auth_prefix() -> str:
    return mlrun.mlconf.secret_stores.kubernetes.auth_secret_name.format(
        hashed_access_key=""
    )


def _new_job_runtime(project: str = "p") -> mlrun.runtimes.KubejobRuntime:
    # Avoid nuclio path; this creates a plain KubejobRuntime without touching files or API
    fn = mlrun.new_function(
        name="f",
        project=project,
        kind="job",
        image="mlrun/mlrun",
    )
    assert hasattr(fn, "set_env"), "Expected runtime to expose set_env"
    return fn


def test_set_env_from_secret_blocks_auth_secret():
    fn = _new_job_runtime()
    forbidden = _auth_prefix() + "xyz"

    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc:
        fn.set_env_from_secret(name="MY_ENV", secret=forbidden)

    assert "Forbidden secret" in str(exc.value)
    assert forbidden in str(exc.value)


def test_set_env_from_secret_allows_regular_secret():
    fn = _new_job_runtime()
    # Should not raise
    fn.set_env_from_secret(name="MY_ENV", secret="regular-secret", secret_key="k")


def test_set_env_blocks_when_value_from_contains_auth_secret_object():
    fn = _new_job_runtime()
    forbidden = _auth_prefix() + "abc"

    value_from = k8s_client.V1EnvVarSource(
        secret_key_ref=k8s_client.V1SecretKeySelector(name=forbidden, key="token")
    )

    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc:
        fn.set_env(name="MY_ENV", value_from=value_from)

    assert "Forbidden secret" in str(exc.value)
    assert forbidden in str(exc.value)


def test_set_env_blocks_when_value_from_contains_auth_secret_dict_variants():
    fn = _new_job_runtime()
    forbidden = _auth_prefix() + "def"

    # CamelCase variant
    value_from_camel = {
        "valueFrom": {
            "secretKeyRef": {
                "name": forbidden,
                "key": "token",
            }
        }
    }

    # snake_case variant
    value_from_snake = {
        "value_from": {
            "secret_key_ref": {
                "name": forbidden,
                "key": "token",
            }
        }
    }

    for payload in (value_from_camel, value_from_snake):
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc:
            fn.set_env(name="MY_ENV", value_from=payload)
        assert "Forbidden secret" in str(exc.value)
        assert forbidden in str(exc.value)


def test_set_env_allows_value_literal_and_non_secret_value_from():
    fn = _new_job_runtime()

    # Plain value should pass
    fn.set_env(name="PLAIN_ENV", value="ok")

    # Non-secret valueFrom (ConfigMap) should also pass
    value_from_config_map = k8s_client.V1EnvVarSource(
        config_map_key_ref=k8s_client.V1ConfigMapKeySelector(
            name="my-configmap", key="cfg"
        )
    )
    fn.set_env(name="FROM_CM", value_from=value_from_config_map)


def test_set_env_blocks_top_level_secret_key_ref_dict():
    fn = _new_job_runtime()
    forbidden = _auth_prefix() + "top"
    payload = {
        "secretKeyRef": {"name": forbidden, "key": "k"},
    }
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        fn.set_env(name="MY_ENV", value_from=payload)
