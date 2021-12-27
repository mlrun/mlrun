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

from mlrun import function_to_module, get_or_create_ctx, new_task
from tests.conftest import examples_path


def test_local_py():
    file_path = f"{examples_path}/training.py"
    mod = function_to_module(file_path)
    task = new_task(inputs={"infile.txt": f"{examples_path}/infile.txt"})
    context = get_or_create_ctx("myfunc", spec=task)
    mod.my_job(context, p1=2, p2="x")
    assert context.results["accuracy"] == 4, "failed to run"
