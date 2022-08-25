from kfp import dsl

import mlrun


# Create a Kubeflow Pipelines pipeline
@dsl.pipeline(name="breast-cancer-demo")
def pipeline(model_name="cancer_classifier"):
    # change to 'keras' to try the 2nd option
    framework = "sklearn"
    if framework == "sklearn":
        serving_class = "mlrun.frameworks.sklearn.SklearnModelServer"
    else:
        serving_class = "mlrun.frameworks.tf_keras.TFKerasModelServer"

    # run the ingestion function with the new image and params
    ingest = mlrun.run_function(
        "get-data",
        name="get-data",
        params={"format": "pq", "model_name": model_name},
        outputs=["dataset"],
    )

    # Train a model
    train = mlrun.run_function(
        "trainer",
        inputs={"dataset": ingest.outputs["dataset"]},
        outputs=["model"],
    )

    # Deploy the trained model as a serverless function
    mlrun.deploy_function(
        "serving",
        models=[
            {
                "key": model_name,
                "model_path": train.outputs["model"],
                "class_name": serving_class,
            }
        ],
    )
