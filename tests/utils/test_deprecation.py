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

import warnings

import deprecated
import pytest

import mlrun


def test_deprecated_decorator_warning_is_shown():
    with warnings.catch_warnings(record=True) as w:

        @deprecated.deprecated(
            version="1.4.0",
            reason="I am deprecated",
            category=FutureWarning,
        )
        def deprecated_function():
            pass

        deprecated_function()

        assert len(w) == 1
        assert issubclass(w[-1].category, FutureWarning)
        assert (
            "Call to deprecated function (or staticmethod) deprecated_function. "
            "(I am deprecated) -- Deprecated since version 1.4.0." in str(w[-1].message)
        )


def test_deprecation_warning_is_shown():
    with warnings.catch_warnings(record=True) as w:
        warnings.warn(
            "I'm a FutureWarning that is raised",
            FutureWarning,
        )

        assert len(w) == 1
        assert issubclass(w[-1].category, FutureWarning)
        assert "I'm a FutureWarning that is raised" in str(w[-1].message)


def test_filter_warnings_decorator():
    warnings.simplefilter("error", FutureWarning)

    @mlrun.utils.filter_warnings("ignore", FutureWarning)
    def warn_ignored():
        warnings.warn("I'm a FutureWarning that is ignored", FutureWarning)

    def warn():
        warnings.warn("I'm a FutureWarning that is raised", FutureWarning)

    # should not raise
    warn_ignored()

    with pytest.raises(FutureWarning):
        warn()


def test_deprecation_helper():
    """
    This test validates that the deprecation warning is shown when using a deprecated class, and that the
    object is created from the new class.
    """
    import mlrun.api.schemas
    import mlrun.common.schemas

    with warnings.catch_warnings(record=True) as w:
        # create an object using the deprecated class
        obj = mlrun.api.schemas.ObjectMetadata(name="name", project="project")

        # validate that the object is created from the new class
        assert type(obj) == mlrun.common.schemas.ObjectMetadata

        # validate that the warning is shown
        assert len(w) == 1
        assert (
            "mlrun.api.schemas.ObjectMetadata is deprecated in version 1.4.0, "
            "Please use mlrun.common.schemas.ObjectMetadata instead."
            in str(w[-1].message)
        )


def test_deprecated_schema_as_argument():
    """
    This test validates that the deprecation warning is shown when using a deprecated schema as an argument to a
    function. And that the function still works, and the schema is converted to the new schema.
    The test uses the get_secrets function as an example.
    """
    import mlrun.api.api.utils
    import mlrun.api.schemas
    import mlrun.common.schemas

    data_session = "some-data-session"

    with warnings.catch_warnings(record=True) as w:
        secrets = mlrun.api.api.utils.get_secrets(
            auth_info=mlrun.api.schemas.AuthInfo(data_session=data_session),
        )

        assert "V3IO_ACCESS_KEY" in secrets
        assert secrets["V3IO_ACCESS_KEY"] == data_session

        # validate that the warning is shown
        assert len(w) == 1
    assert (
        "mlrun.api.schemas.AuthInfo is deprecated in version 1.4.0, "
        "Please use mlrun.common.schemas.AuthInfo instead." in str(w[-1].message)
    )
