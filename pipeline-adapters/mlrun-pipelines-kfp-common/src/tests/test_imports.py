# Copyright 2025 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import mlrun_pipelines.common.imports


def test_dummy_compiler_calls():
    container = mlrun_pipelines.common.imports.DummyCompiler()
    expected_compiler = container.Compiler()
    result_compiler = expected_compiler()
    assert expected_compiler == result_compiler
    result_compiler.compile(a="test")
