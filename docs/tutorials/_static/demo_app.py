import pandas as pd

import mlrun
from mlrun.artifacts import (
    TableArtifact,
)
from mlrun.common.schemas.model_monitoring.constants import (
    ResultKindApp,
    ResultStatusApp,
)
from mlrun.model_monitoring.application import (
    ModelMonitoringApplicationBase,
    ModelMonitoringApplicationResult,
)


class DemoMonitoringApp(ModelMonitoringApplicationBase):
    NAME = "monitoring-test"
    check_num_events = True

    def do_tracking(
        self,
        application_name: str,
        sample_df_stats: mlrun.common.model_monitoring.helpers.FeatureStats,
        feature_stats: mlrun.common.model_monitoring.helpers.FeatureStats,
        sample_df: pd.DataFrame,
        start_infer_time: pd.Timestamp,
        end_infer_time: pd.Timestamp,
        latest_request: pd.Timestamp,
        endpoint_id: str,
        output_stream_uri: str,
    ) -> list[ModelMonitoringApplicationResult]:
        self.context.logger.info("Running demo app")
        self.context.log_artifact(
            TableArtifact(f"sample_df_{start_infer_time}", df=sample_df)
        )
        return [
            ModelMonitoringApplicationResult(
                name="data_drift_test",
                value=2.15,
                kind=ResultKindApp.data_drift,
                status=ResultStatusApp.detected,
            ),
            ModelMonitoringApplicationResult(
                name="model_perf",
                value=80,
                kind=ResultKindApp.model_performance,
                status=ResultStatusApp.no_detection,
            ),
        ]
