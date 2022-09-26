(test_and_deploy_batch_model_inference)=

# Test and deploy batch model inference

## Testing model

To evaluate batch model inference, you should use the evaluate handler of the `auto_trainer` function.

This is typically done during model development, for more information refer to the {ref}`auto_trainer_evaluate` handler documentation. For example:

``` python
import mlrun

# Set the base project name
project_name_base = 'batch-inference'

# Initialize the MLRun project object
project = mlrun.get_or_create_project(project_name_base, context="./", user_project=True)

auto_trainer = mlrun.import_function(mlrun.import_function("hub://auto_trainer"))

evaluate_run = project.run_function(
    auto_trainer,
    handler="evaluate",
    inputs={"dataset": train_run.outputs['test_set']},
    params={
        "model": train_run.outputs['model'],
        "label_columns": "labels",
    },
)
```

## Deploy model

Batch inference is implemented in MLRun by running the function with an input dataset. With MLRun one can easily create any custom logic in a function, including loading a model and calling it.

One you create the function, you can call {ref}`schedule it<scheduled-jobs>` or run it as part of a {ref}`workflow<workflow-overview>`.

It is recommended you use the {ref}`built-in batch inference function<using_built-in_batch_inference>` if you are using one of the frameworks supported by MLRun, which includes not just inference, but also drift analysis.
