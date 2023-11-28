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

#This is a wrapper class for the mlrun to interact with the huggingface's evaluate library
import uuid
from typing import Union, List
from mlrun.model_monitoring.application import ModelMonitoringApplication, ModelMonitoringApplicationResult

_HAS_evaluate = False
try:
    import evaluate # noqa: F401
    _HAS_evaluate = True
except ModuleNotFoundError:
    pass

if _HAS_evaluate:
    import evaluate

class HFEvaluateApplication(ModelMonitoringApplication):
    def __init__(self, model,  metrics: List[str], metrics_params: Dict[str, Any] = None): 
        # Since it's a customize class, it's running as a job, need to take care of the whole thing
        # Model we want to evaluate, currently we have sample_df, where we can get the y_hat and y_true
        # The basic idea is to use the evaluate library to evaluate the hf model
        """
        A class for integrating evaluate for mlrun model monitoring within a monitoring application.
        Note: evaluate is not installed by default and must be installed separately.
        """
        if not _HAS_evaluate:
            raise ModuleNotFoundError("evaluate is not installed - the app cannot run")

        self.model = model
        self.metrics_params = metrics_params
        self.metrics = metrics

    def run_application(
            self,
            application_name: str,
            sample_df_stats: pd.DataFrame,
            feature_stats: pd.DataFrame,
            sample_df: pd.DataFrame,
            schedule_time: pd.Timestamp,
            latest_request: pd.Timestamp,
            endpoint_id: str,
            output_stream_uri: str,
        ) -> Union[
            ModelMonitoringApplicationResult, List[ModelMonitoringApplicationResult]
        ]:

        """
        Run the application
        :param application_name: (str) The name of the application
        :param sample_df_stats: (pd.DataFrame) The statistics of the sample dataframe
        :param feature_stats: (pd.DataFrame) The statistics of the features
        :param sample_df: (pd.DataFrame) The sample dataframe
        :param schedule_time: (pd.Timestamp) The time of the scheduled run
        :param latest_request: (pd.Timestamp) The time of the latest request
        :param endpoint_id: (str) The ID of the endpoint
        :param output_stream_uri: (str) The URI of the output stream
        :return: (Union[ModelMonitoringApplicationResult, List[ModelMonitoringApplicationResult]]) The result of the application
        """
        self.context.logger.info("Running evaluate app")


        # Evaluate the model and log the metrics
        y_hat = self.model.predict(sample_df).tolist()
        # where to store the y_hat if we still need the result
        y_true = sample_df["label"].tolist()

        
        metrics = evaluate.combine(self.metrics)
        result = metrics.compute(predictions=y_hat, references=y_true, **self.metrics_params)
        self.result = result
        self.context.logger.log_artifact("metrics", body=result)
 

    def log_radar_plot(self, artifact_name: str):
        """
        Log the radar plot to the MLRun context
        :param artifact_name: (str) The name of the artifact to log
        """

        self.context.logger.log_artifact(artifact_name, body=evaluate.visualization.radar_plot(self.result))
