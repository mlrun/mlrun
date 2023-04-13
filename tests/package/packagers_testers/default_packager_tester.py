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
import os
import sys
import tempfile
from typing import Tuple

import cloudpickle

from mlrun.package import DefaultPackager
from tests.package.packager_tester import (
    PackagerTester,
    PackTest,
    PackToUnpackTest,
    UnpackTest,
)


class SomeClass:
    def __init__(self, a: int, b: int, c: int):
        self.a = a
        self.b = b
        self.c = c

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b and self.c == other.c

    def __str__(self):
        return str(self.a + self.b + self.c)


def pack_some_class() -> SomeClass:
    return SomeClass(a=1, b=2, c=3)


def unpack_some_class(obj: SomeClass):
    assert type(obj).__name__ == SomeClass.__name__
    assert obj == SomeClass(a=1, b=2, c=3)


def validate_some_class_result(result: str) -> bool:
    return result == "6"


def pickle_some_class() -> Tuple[str, str]:
    temp_path = tempfile.mkdtemp()
    pkl_path = os.path.join(temp_path, "my_class.pkl")
    some_class = SomeClass(a=1, b=2, c=3)
    with open(pkl_path, "wb") as pkl_file:
        cloudpickle.dump(some_class, pkl_file)

    return temp_path, pkl_path


class DefaultPackagerTester(PackagerTester):
    PACKAGER_IN_TEST = DefaultPackager

    TESTS = [
        PackTest(
            pack_handler="pack_some_class",
            parameters={},
            log_hint="my_result : result",
            validation_function=validate_some_class_result,
        ),
        UnpackTest(
            prepare_input_function=pickle_some_class,
            unpack_handler="unpack_some_class",
        ),
        PackToUnpackTest(
            pack_handler="pack_some_class",
            parameters={},
            log_hint="my_object",
            expected_instructions={
                "pickle_module_name": "cloudpickle",
                "pickle_module_version": cloudpickle.__version__,
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            },
            unpack_handler="unpack_some_class",
        ),
    ]
