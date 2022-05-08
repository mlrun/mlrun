import mlrun
from kfp import dsl

# Create a Kubeflow Pipelines pipeline
@dsl.pipeline(name="breast-cancer-demo")
def pipeline(model_name="cancer_classifier"):
    # run the ingestion function with the new image and params
    ingest = mlrun.run_function(
        "gen-breast-cancer",
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
    deploy = mlrun.deploy_function(
        "serving",
        models=[{"key": model_name, "model_path": train.outputs["model"], "class_name": 'ClassifierModel'}],
    )
