import mlrun.model_monitoring.applications as mm_applications


class DemoMonitoringAppV2(mm_applications.ModelMonitoringApplicationBaseV2):
    _dict_fields = ["param_1", "param_2"]

    def __init__(self, param_1, **kwargs) -> None:
        self.param_1 = param_1
        self.param_2 = kwargs["param_2"]

    def do_tracking(
        self,
        monitoring_context,
    ):
        pass
