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
import pytest

from mlrun.k8s_utils import sanitize_label_value


@pytest.mark.parametrize(
    "value, expected",
    [
        ("my-value", "my-value"),
        ("foo%bar", "foo-bar"),
        (
            "very{long}[string](value)#with#$several$|illegal|;characters;'present'",
            "very-long--string--value--with--several--illegal--characters--p",
        ),
        ("0.0.0+unstable", "0.0.0-unstable"),
    ],
)
def test_sanitize_label_value(value: str, expected: str):
    assert sanitize_label_value(value) == expected
