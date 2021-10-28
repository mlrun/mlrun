import mlrun
import mlrun.builder
import mlrun.api.schemas
import unittest.mock


def test_build_runtime_use_base_image_when_no_build():
    fn = mlrun.new_function("some-function", "some-project", "some-tag", kind="job")
    base_image = "mlrun/ml-models"
    fn.build_config(base_image=base_image)
    assert fn.spec.image == ""
    mlrun.builder.build_image = unittest.mock.Mock(return_value="skipped")
    ready = mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(),
        fn,
        with_mlrun=True,
        mlrun_version_specifier=None,
        skip_deployed=False,
        builder_env=None,
    )
    assert ready is True
    assert fn.spec.image == base_image
