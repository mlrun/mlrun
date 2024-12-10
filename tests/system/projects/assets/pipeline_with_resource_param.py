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
import kfp

import mlrun


@kfp.dsl.pipeline(
    name="Demo passing param to function spec", description="Shows how to use mlrun."
)
def kfpipeline(memory: str = "10Mi"):
    time_to_sleep = 2
    project: mlrun.projects.MlrunProject = mlrun.get_current_project()
    func: mlrun.runtimes.KubejobRuntime = project.get_function("func-1")
    func.with_requests(mem=str(memory))
    mlrun.run_function(
        func,
        params={"time_to_sleep": time_to_sleep},
        outputs=["return"],
    )
