from kfp import dsl

import mlrun


# Create a Kubeflow Pipelines pipeline
@dsl.pipeline(name="breast-cancer-demo")
def pipeline(model_name="cancer-classifier"):
    # run the ingestion function with the new image and params
    ingest = mlrun.run_function(
        "data-prep",
        name="get-data",
        params={"format": "pq", "model_name": model_name},
        outputs=["dataset"],
    )

    # Train a model using the auto_trainer hub function
    train = mlrun.run_function(
        "hub://auto_trainer",
        inputs={"dataset": ingest.outputs["dataset"]},
        params={
            "model_class": "sklearn.ensemble.RandomForestClassifier",
            "train_test_split_size": 0.2,
            "label_columns": "label",
            "model_name": "cancer",
        },
        handler="train",
        outputs=["model"],
    )

    # Deploy the trained model as a serverless function
    serving_fn = mlrun.new_function("serving", image="mlrun/mlrun", kind="serving")
    serving_fn.with_code(body=" ")
    mlrun.deploy_function(
        serving_fn,
        models=[
            {
                "key": model_name,
                "model_path": train.outputs["model"],
                "class_name": "mlrun.frameworks.sklearn.SklearnModelServer",
            }
        ],
    )
