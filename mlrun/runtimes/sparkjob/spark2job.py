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

from mlrun.config import config

from ...utils import update_in
from .abstract import AbstractSparkJobSpec, AbstractSparkRuntime


class Spark2JobSpec(AbstractSparkJobSpec):
    pass


class Spark2Runtime(AbstractSparkRuntime):
    def _enrich_job(self, job):
        update_in(job, "spec.serviceAccount", "sparkapp")
        return

    def with_priority_class(
        self, name: str = config.default_function_priority_class_name
    ):
        raise NotImplementedError("Not supported in spark 2 operator")

    def _get_spark_version(self):
        return "2.4.5"

    def _get_igz_deps(self):
        return {
            "jars": [
                "/spark/v3io-libs/v3io-hcfs_2.11.jar",
                "/spark/v3io-libs/v3io-spark2-streaming_2.11.jar",
                "/spark/v3io-libs/v3io-spark2-object-dataframe_2.11.jar",
                "/igz/java/libs/scala-library-2.11.12.jar",
            ],
            "files": ["/igz/java/libs/v3io-pyspark.zip"],
        }

    @property
    def spec(self) -> Spark2JobSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", Spark2JobSpec)
