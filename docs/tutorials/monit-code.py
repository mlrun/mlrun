import re
from typing import Any, Union

import mlrun
import mlrun.common.schemas
from mlrun.model_monitoring.applications import (
    ModelMonitoringApplicationBase,
    ModelMonitoringApplicationResult,
)

STATUS_RESULT_MAPPING = {
    0: mlrun.common.schemas.model_monitoring.constants.ResultStatusApp.detected,
    1: mlrun.common.schemas.model_monitoring.constants.ResultStatusApp.no_detection,
}


class LLMMonitoringFunction(ModelMonitoringApplicationBase):

    def do_tracking(
        self,
        monitoring_context,
    ) -> Union[
        ModelMonitoringApplicationResult, list[ModelMonitoringApplicationResult]
    ]:
        
        # User monitoring sampling, in this case an integer representing model performance
        # Can be calulated based off the traffic to the function using monitoring_context.sample_df
        result = 0.9

        monitoring_context.log_dataset(
            key="llm-monitoring-df",
            df=monitoring_context.sample_df
        )

        # get status:
        status = STATUS_RESULT_MAPPING[round(result)]

        return ModelMonitoringApplicationResult(
            name="llm_monitoring_df",
            value=result,
            kind=mlrun.common.schemas.model_monitoring.constants.ResultKindApp.model_performance,
            status=status,
            extra_data={},
        )
