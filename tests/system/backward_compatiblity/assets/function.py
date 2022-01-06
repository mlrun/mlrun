from pickle import dumps

import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

import mlrun
from mlrun.artifacts import ChartArtifact
from mlrun.mlutils.plots import eval_model_v2


def backward_compatibility_test_in_runtime_success(context: mlrun.MLClientCtx):
    # Simple artifact logging
    context.log_artifact(
        "model",
        body=b"abc is 123",
        local_path="model.txt",
        labels={"framework": "xgboost"},
    )
    # logging ChartArtifact
    chart = ChartArtifact("chart")
    chart.labels = {"type": "roc"}
    chart.header = ["Epoch", "Accuracy", "Loss"]
    for i in range(1, 8):
        chart.add_row([i, i / 20 + 0.75, 0.30 - i / 20])
    context.log_artifact(chart)

    # DataSet logging
    raw_data = {
        "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
        "last_name": ["Miller", "Jacobson", "Ali", "Milner", "Cooze"],
        "age": [42, 52, 36, 24, 73],
        "testScore": [25, 94, 57, 62, 70],
    }
    df = pd.DataFrame(raw_data, columns=["first_name", "last_name", "age", "testScore"])
    context.log_dataset("mydf", df=df, stats=True)

    # Model logging
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = linear_model.LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    eval_metrics = eval_model_v2(context, X_test, y_test, model)
    context.log_model(
        "model",
        body=dumps(model),
        artifact_path=context.artifact_subpath("models"),
        extra_data=eval_metrics,
        model_file="model.pkl",
        metrics=context.results,
        labels={"class": "sklearn.linear_model.LogisticRegression"},
    )


def backward_compatibility_failure_in_running_job():
    raise RuntimeError("Failing on purpose")
