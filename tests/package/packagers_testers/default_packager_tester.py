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
from mlrun.package import LogHintKey

from ..packager_tester import PackagerTester, PackTest, PackToUnpackTest, UnpackTest


class SomeClass:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b and self.c == other.c

    def __str__(self):
        return self.a + self.b + self.c


def pack_some_class() -> SomeClass:
    return SomeClass(a=1, b=2, c=3)


def unpack_some_class(obj: SomeClass):
    assert isinstance(obj, SomeClass)
    assert obj == SomeClass(a=1, b=2, c=3)


def validate_some_class_result(result: int) -> bool:
    return result == 6


class DefaultPackagerTester(PackagerTester):
    TESTS = [
        PackTest(
            pack_handler="pack_some_class",
            parameters={},
            log_hint={LogHintKey.KEY: "my_result", LogHintKey.ARTIFACT_TYPE: "result"},
            validation_function=validate_some_class_result,
        ),
        PackToUnpackTest(
            pack_handler="pack_some_class",
            parameters={},
            log_hint={LogHintKey.KEY: "my_object"},
            expected_instructions={},
            unpack_handler="unpack_some_class",
        ),
    ]
