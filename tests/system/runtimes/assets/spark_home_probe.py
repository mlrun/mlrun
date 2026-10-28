# Copyright 2026 Iguazio
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
import os
import site

from pyspark.sql import SparkSession

import mlrun


def probe(context):
    spark = SparkSession.builder.appName("non-root-home-probe").getOrCreate()
    spark.stop()
    # an install into the image user's site is hidden when HOME is replaced (ML-13151)
    context.log_results(
        {
            "uid": os.getuid(),
            "home": os.environ.get("HOME"),
            "user_site": site.getusersitepackages(),
            "mlrun_path": mlrun.__file__,
        }
    )
