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
import os
import tempfile
from typing import Tuple

import cloudpickle

from mlrun.package import DefaultPackager
from tests.package.packager_tester import (
    COMMON_OBJECT_INSTRUCTIONS,
    NewClass,
    PackagerTester,
    PackTest,
    PackToUnpackTest,
    UnpackTest,
)


def pack_some_class() -> NewClass:
    return NewClass(a=1, b=2, c=3)


def unpack_some_class(obj: NewClass):
    assert type(obj).__name__ == NewClass.__name__
    assert obj == NewClass(a=1, b=2, c=3)


def validate_some_class_result(result: str) -> bool:
    return result == "6"


def prepare_new_class() -> Tuple[str, str]:
    temp_directory = tempfile.mkdtemp()
    pkl_path = os.path.join(temp_directory, "my_class.pkl")
    some_class = NewClass(a=1, b=2, c=3)
    with open(pkl_path, "wb") as pkl_file:
        cloudpickle.dump(some_class, pkl_file)

    return pkl_path, temp_directory


class DefaultPackagerTester(PackagerTester):
    """
    A tester for the `DefaultPackager`.
    """

    PACKAGER_IN_TEST = DefaultPackager

    TESTS = [
        PackTest(
            pack_handler="pack_some_class",
            log_hint="my_result : result",
            validation_function=validate_some_class_result,
        ),
        UnpackTest(
            prepare_input_function=prepare_new_class,
            unpack_handler="unpack_some_class",
        ),
        PackToUnpackTest(
            pack_handler="pack_some_class",
            log_hint="my_object",
            expected_instructions={
                "object_module_name": "tests",
                **COMMON_OBJECT_INSTRUCTIONS,
            },
            unpack_handler="unpack_some_class",
        ),
    ]
