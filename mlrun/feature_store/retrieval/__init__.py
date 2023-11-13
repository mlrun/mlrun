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
import mlrun.errors

from .dask_merger import DaskFeatureMerger
from .job import RemoteVectorResponse, run_merge_job  # noqa
from .local_merger import LocalFeatureMerger
from .spark_merger import SparkFeatureMerger
from .storey_merger import StoreyFeatureMerger

mergers = {
    "local": LocalFeatureMerger,
    "dask": DaskFeatureMerger,
    "spark": SparkFeatureMerger,
    "storey": StoreyFeatureMerger,
}


def get_merger(kind):
    if not kind:
        return LocalFeatureMerger
    merger = mergers.get(kind)
    if not merger:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"No merger was found for engine '{kind}'. Supported engines: {', '.join(mergers)}."
        )
    return merger
