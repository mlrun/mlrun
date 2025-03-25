import mlrun.model_monitoring.applications.context as mm_context
import mlrun.model_monitoring.applications.results as mm_results
from mlrun.artifacts import (
    TableArtifact,
)
from mlrun.common.schemas.model_monitoring.constants import (
    ResultKindApp,
    ResultStatusApp,
)
from mlrun.model_monitoring.applications import (
    ModelMonitoringApplicationBase,
    ModelMonitoringApplicationResult,
)


class DemoMonitoringApp(ModelMonitoringApplicationBase):
    NAME = "monitoring-test"
    check_num_events = True

    def do_tracking(
        self, monitoring_context: mm_context.MonitoringApplicationContext
    ) -> list[mm_results.ModelMonitoringApplicationResult]:
        monitoring_context.logger.info("Running demo app")
        monitoring_context.nuclio_logger.info("Running demo app")
        # The artifact key validation is preventing the inclusion of the characters "+", ":", and spaces.
        formatted_start_infer_time = monitoring_context.start_infer_time.strftime(
            "%Y-%m-%dT%H_%M_%S%z"
        ).replace("+", "P")
        monitoring_context.log_artifact(
            TableArtifact(
                f"sample_df_{formatted_start_infer_time}",
                df=monitoring_context.sample_df,
            )
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
