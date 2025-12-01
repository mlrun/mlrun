from typing import Union

import pandas as pd

import mlrun.model_monitoring.applications.context as mm_context
from mlrun.common.schemas.model_monitoring.constants import (
    ResultKindApp,
    ResultStatusApp,
)
from mlrun.model_monitoring.applications import (
    ModelMonitoringApplicationBase,
    ModelMonitoringApplicationMetric,
    ModelMonitoringApplicationResult,
)


class ModelMonitoringApplication(ModelMonitoringApplicationBase):
    name = "LLModelMonitoringApplication"

    def do_tracking(
        self,
        monitoring_context: mm_context.MonitoringApplicationContext,
    ) -> list[
        Union[ModelMonitoringApplicationResult, ModelMonitoringApplicationMetric]
    ]:
        """"""
        df = monitoring_context.sample_df
        if df.empty:
            monitoring_context.logger.warning(
                "Empty dataframe received, skipping tracking"
            )
            return [], []
        # Example of processing the dataframe and creating results
        results = []
        metrics = []
        df = self._update_dataframe_with_usage(df)
        # Calculate max, min, avg, and std for the 'usage'
        for column in ["completion_tokens", "prompt_tokens", "total_tokens"]:
            if column in df.columns:
                stats = self._calculate_max_min_avg_std(df, column)
                results.append(
                    self._create_result(
                        name=f"{column}_stats",
                        value=stats["avg"],
                        kind=ResultKindApp.model_performance,
                        threshold=150,  # Example threshold
                    )
                )
                metrics.append(
                    self._create_metric(
                        name=f"{column}_max",
                        value=stats["max"],
                    )
                )
                metrics.append(
                    self._create_metric(
                        name=f"{column}_min",
                        value=stats["min"],
                    )
                )
                metrics.append(
                    self._create_metric(
                        name=f"{column}_std",
                        value=stats["std"],
                    )
                )
        return results + metrics

    @staticmethod
    def _update_dataframe_with_usage(df: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [df.drop(columns="usage"), df["usage"].apply(pd.Series)], axis=1
        )

    @staticmethod
    def _calculate_max_min_avg_std(df: pd.DataFrame, column: str) -> dict:
        """
        Calculate max, min, avg, and std for a given column in the dataframe.
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataframe.")

        return {
            "max": df[column].max(),
            "min": df[column].min(),
            "avg": df[column].mean(),
            "std": df[column].std(),
        }

    @staticmethod
    def _create_result(
        name: str,
        value: float,
        kind: ResultKindApp,
        threshold: float,
    ) -> ModelMonitoringApplicationResult:
        status = ResultStatusApp.no_detection
        if value > threshold:
            status = ResultStatusApp.detected
        return ModelMonitoringApplicationResult(
            name=name,
            value=value,
            kind=kind,
            status=status,
            extra_data={
                "threshold": threshold,
                "value": value,
            },
        )

    @staticmethod
    def _create_metric(
        name: str,
        value: float,
    ) -> ModelMonitoringApplicationMetric:
        return ModelMonitoringApplicationMetric(
            name=name,
            value=value,
        )
