(build-run-workflows-pipelines)=
# Build and run workflows/pipelines

This section shows how to write a batch pipeline so that it can be executed via an MLRun Project.
With a batch pipeline, you can use the MLRun Project to execute several Functions in a DAG using the Python SDK or CLI.

This example creates a project with three MLRun functions and a single pipeline that orchestrates them. The pipeline steps are:
- `get-data` &mdash; Get iris data from sklearn
- `train-model` &mdash; Train model via sklearn
- `deploy-model` &mdash; Deploy model to HTTP endpoint  

```
import mlrun
project = mlrun.get_or_create_project(\"iguazio-academy\", context=\"./\")
```

## Add functions to a project
   
Add the functions to a project:

```
project.set_function(name='get-data', func='functions/get_data.py', kind='job', image='mlrun/mlrun')
project.set_function(name='train-model', func='functions/train.py', kind='job', image='mlrun/mlrun'),
project.set_function(name='deploy-model', func='hub://v2_model_server')
```

## Write a pipeline

Next, define the pipeline that orchestrates the three components. This pipeline is simple, however you can create very complex pipelines with branches, conditions, and more.

```
%%writefile pipelines/training_pipeline.py
from kfp import dsl
import mlrun

@dsl.pipeline(
    name=\"batch-pipeline-academy\",
    description=\"Example of batch pipeline for Iguazio Academy\"
)
def pipeline(label_column: str, test_size=0.2):
    
    # Ingest the data set
    ingest = mlrun.run_function(
        'get-data',
        handler='prep_data',
        params={'label_column': label_column},
        outputs=[\"iris_dataset\"]
    )
    
    # Train a model   
    train = mlrun.run_function(
        \"train-model\",
        handler=\"train_model\",
        inputs={\"dataset\": ingest.outputs[\"iris_dataset\"]},
        params={
            \"label_column\": label_column,
            \"test_size\" : test_size
        },
        outputs=['model']
    )
    
    # Deploy the model as a serverless function
    deploy = mlrun.deploy_function(
        \"deploy-model\",
        models=[{\"key\": \"model\", \"model_path\": train.outputs[\"model\"]}]
    )
```

## Add a pipeline to a project

Add the pipeline to your project:

```
project.set_workflow(name='train', workflow_path=\"pipelines/training_pipeline.py")
project.save()
``` 