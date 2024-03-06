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


from abc import ABC

from mlrun.common.schemas.model_monitoring import (
    AppResultEvent,
)


class TSDBstore(ABC):
    def __init__(self, project: str):
        """
        Initialize a new TSDB store target.
        :param project:             The name of the project.
        """
        self.project = project

    def apply_monitoring_stream_steps(self, graph, **kwargs):
        """
        Apply TSDB steps on the provided monitoring graph. Throughout these steps, the graph stores live data of
        different key metric dictionaries in TSDB target. This data is being used by the monitoring dashboards in
        grafana.
        There are 3 different key metric dictionaries that are being generated throughout these steps:
        - base_metrics (average latency and predictions over time)
        - endpoint_features (Prediction and feature names and values)
        - custom_metrics (user-defined metrics)
        """
        pass

    def write_application_event(self, event: AppResultEvent):
        """
        Write a single application result event to the TSDB target.
        """
        pass

    def update_default_data_drift(self, **kwargs):
        """
        Update drift results in input stream and TSDB table. The drift results within the input stream are stored
         only if the result indicates on possible drift (or detected drift).
        """
        pass

    def delete_tsdb_resources(self, **kwargs):
        pass

    def get_endpoint_real_time_metrics(self, **kwargs):
        """
        Getting metrics from the time series DB. There are pre-defined metrics for model endpoints such as
        `predictions_per_second` and `latency_avg_5m` but also custom metrics defined by the user.
        """
        pass
