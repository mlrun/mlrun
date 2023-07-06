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
import kfp.dsl

import mlrun


def func1(context, p1=1):
    context.log_result("accuracy", p1 * 2)


@kfp.dsl.pipeline(name="remote_pipeline", description="tests remote pipeline")
def pipeline():
    run1 = mlrun.run_function("func1", handler="func1", params={"p1": 9})
    print(run1)
    run2 = mlrun.run_function("func2", handler="func1", params={"p1": 29})
    print(run2)
    build3 = mlrun.build_function("func3")
    print(build3)
    dep4 = mlrun.deploy_function("func4")
    print(dep4)
