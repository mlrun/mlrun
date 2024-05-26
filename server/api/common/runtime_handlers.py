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
import copy

from mlrun import mlconf
from mlrun.common.constants import MLRunInternalLabels


def get_resource_labels(function, run=None, scrape_metrics=None):
    scrape_metrics = (
        scrape_metrics if scrape_metrics is not None else mlconf.scrape_metrics
    )
    run_uid, run_name, run_project, run_owner = None, None, None, None
    if run:
        run_uid = run.metadata.uid
        run_name = run.metadata.name
        run_project = run.metadata.project
        run_owner = run.metadata.labels.get(MLRunInternalLabels.owner)
    labels = copy.deepcopy(function.metadata.labels)
    labels[MLRunInternalLabels.mlrun_class] = function.kind
    labels[MLRunInternalLabels.project] = run_project or function.metadata.project
    labels[MLRunInternalLabels.function] = str(function.metadata.name)
    labels[MLRunInternalLabels.tag] = str(function.metadata.tag or "latest")
    labels[MLRunInternalLabels.scrape_metrics] = str(scrape_metrics)

    if run_uid:
        labels[MLRunInternalLabels.uid] = run_uid

    if run_name:
        labels[MLRunInternalLabels.name] = run_name

    if run_owner:
        labels[MLRunInternalLabels.mlrun_owner] = run_owner
        if "@" in run_owner:
            run_owner, domain = run_owner.split("@")
            labels[MLRunInternalLabels.mlrun_owner] = run_owner
            labels[MLRunInternalLabels.owner_domain] = domain

    return labels
