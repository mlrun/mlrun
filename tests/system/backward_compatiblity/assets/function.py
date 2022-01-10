import pandas as pd

import mlrun
from mlrun.artifacts import ChartArtifact


def api_backward_compatibility_tests_succeeding_function(context: mlrun.MLClientCtx):
    # Simple artifact logging
    logged_artifact = context.log_artifact(
        "model",
        body=b"abc is 123",
        local_path="model.txt",
        labels={"framework": "xgboost"},
    )
    context.logger.info("Logged artifact", artifact=logged_artifact.base_dict())
    artifact = context.get_store_resource(logged_artifact.uri)
    context.logger.info("Got artifact", artifact=artifact.uri)

    # logging ChartArtifact
    chart = ChartArtifact("chart")
    chart.labels = {"type": "roc"}
    chart.header = ["Epoch", "Accuracy", "Loss"]
    for i in range(1, 8):
        chart.add_row([i, i / 20 + 0.75, 0.30 - i / 20])
    logged_chart = context.log_artifact(chart)
    context.logger.info(
        "Logged chart artifact", chart_artifact=logged_chart.base_dict()
    )

    # DataSet logging
    raw_data = {
        "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
    }
    df = pd.DataFrame(raw_data, columns=["first_name"])
    logged_dataset = context.log_dataset("mydf", df=df, stats=True)
    context.logger.info("Logged dataset", dataset_artifact=logged_dataset.base_dict())

    # Model logging
    logged_model = context.log_model(
        "model",
        body="{}",
        artifact_path=context.artifact_subpath("models"),
        model_file="model.pkl",
        labels={"type": "test"},
    )
    context.logger.info("Logged model", model_artifact=logged_model.base_dict())


def api_backward_compatibility_tests_failing_function():
    raise RuntimeError("Failing on purpose")
