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

from .dask_merger import DaskFeatureMerger
from .job import run_merge_job  # noqa
from .local_merger import LocalFeatureMerger
from .online import init_feature_vector_graph  # noqa
from .spark_merger import SparkFeatureMerger

mergers = {
    "local": LocalFeatureMerger,
    "dask": DaskFeatureMerger,
    "spark": SparkFeatureMerger,
}


def get_merger(kind):
    if not kind:
        return LocalFeatureMerger
    return mergers.get(kind)
