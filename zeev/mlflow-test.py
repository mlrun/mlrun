
import pandas as pd
from sklearn.datasets import load_breast_cancer
import os
import sys

# List of servers:
SERVERS = [  # [0] = Server's suffix | [1] = User name | [2] = Access key
    ("cto-office.iguazio-cd1.com", "zeevr", "6dcf8f3a-3355-4a57-86de-f59757044dae"),  # 0
]

IS_ONLINE = True

# Chosen server:
SERVER_INDEX = 0
if IS_ONLINE:
    # Setup environment to use MLRun in the chosen server:
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["MLRUN_DBPATH"] = "https://mlrun-api.default-tenant.app.{}".format(
        SERVERS[SERVER_INDEX][0]
    )
    os.environ["MLRUN_ARTIFACT_PATH"] = "/User/artifacts/{{project}}/{{run.uid}}"
    os.environ["V3IO_USERNAME"] = SERVERS[SERVER_INDEX][1]
    os.environ["V3IO_API"] = "https://webapi.default-tenant.app.{}".format(
        SERVERS[SERVER_INDEX][0]
    )
    os.environ["V3IO_ACCESS_KEY"] = SERVERS[SERVER_INDEX][2]


import mlrun


def main():
    project = mlrun.get_or_create_project("mlflow-test1", context="./", user_project=True)
    trainer = project.set_function("zeev/train.py", name="train", kind="job", image="mlrun/mlrun", handler="train")
    trainer.spec.build.commands = ["python -m pip install mlflow"]
    trainer.save()
    trainer.deploy()
    # trainer = mlrun.code_to_function(name='trainer', filename='zeev/train.py', kind="job", image="mlrun/mlrun",
    #                                  handler="train", requirements=['mlflow'])

    breast_cancer = load_breast_cancer()
    breast_cancer_dataset = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
    breast_cancer_labels = pd.DataFrame(data=breast_cancer.target, columns=["label"])
    breast_cancer_dataset = pd.concat([breast_cancer_dataset, breast_cancer_labels], axis=1)

    breast_cancer_dataset.to_csv("./cancer-dataset.csv", index=False)

    # trainer_run = trainer.run(
    #     inputs={"dataset": os.path.abspath("./cancer-dataset.csv")},
    #     params={"n_estimators": 100, "learning_rate": 1e-1, "max_depth": 3},
    #     local=True,
    #     artifact_path='./'
    # )
    trainer_run = trainer.run(
        inputs={"dataset": os.path.abspath("./cancer-dataset.csv")},
        params={"n_estimators": 100, "learning_rate": 1e-1, "max_depth": 3},
        local=False
    )

if __name__ == "__main__":
    main()