from mlrun.execution import MLClientCtx


def sample(context: MLClientCtx, metric_name: str):
    # Get the metric value from the application monitoring

    prj = context.get_project_object()
    alert = prj.list_alerts_configs()[0]

    # Check is alert was triggered
    if alert.state == "active":
        context.log_result("alert_triggered", "True")
        prj.reset_alert_config(alert_name=metric_name)
    else:
        context.log_result("alert_triggered", "False")
