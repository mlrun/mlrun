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
import mlrun


def func1(context, p1=1):
    context.log_result("accuracy", p1 * 2)


def func2(context, x=0):
    context.log_result("y", x + 1)


def func3(context, p1=0, p2=None):
    context.log_result("p1", p1)
    context.log_result("p2", p2)


def my_pipe(param1=0):
    run1 = mlrun.run_function("tstfunc", handler="func1", params={"p1": param1})
    print(run1.to_yaml())

    run2 = mlrun.run_function(
        "tstfunc", handler="func2", params={"x": run1.outputs["accuracy"]}
    )
    print(run2.to_yaml())

    # hack to return run result to the test for assertions
    mlrun.projects.pipeline_context._test_result = run2
    mlrun.projects.pipeline_context._artifact_path = (
        mlrun.projects.pipeline_context.workflow_artifact_path
    )


def args_pipe(param1=0, param2=None):
    run = mlrun.run_function(
        "tstfunc", handler="func3", params={"p1": param1, "p2": param2}
    )
    mlrun.projects.pipeline_context._test_result = run
