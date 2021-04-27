# MLRun CI Example
# ================
# this code can run in the IDE or inside a CI/CD script (Github Actions or Gitlab CI/CD)
# and require setting the following env vars (can be done in the CI system):
#
#   MLRUN_DBPATH               - url of the mlrun cluster
#   V3IO_USERNAME              - username in the remote cluster
#   V3IO_ACCESS_KEY            - access key to the remote cluster
#   GIT_TOKEN or GITHUB_TOKEN  - Github/Gitlab API Token (will be set automatically in Github Actions)
#   SLACK_WEBHOOK              - optional, Slack API key when using slack notifications
#

import json
from mlrun.utils import RunNotifications
import mlrun
from mlrun.platforms import auto_mount

project = "ci"

# create notification object (console, Git, Slack as outputs) and push start message
notifier = RunNotifications(with_slack=True).print()
# use the following line only when running inside Github actions or Gitlab CI
notifier.git_comment()

notifier.push_start_message(project)

# define and run a local data prep function
data_prep_func = mlrun.code_to_function("prep-data", filename="../scratch/prep_data.py", kind="job",
                                        image="mlrun/mlrun", handler="prep_data").apply(auto_mount())

# Set the source-data URL
source_url = 'https://s3.wasabisys.com/iguazio/data/iris/iris.data.raw.csv'
prep_data_run = data_prep_func.run(name='prep_data', inputs={'source_url': source_url})

# train the model using a library (hub://) function and the generated data
train = mlrun.import_function('hub://sklearn_classifier').apply(auto_mount())
train_run = train.run(name='train',
                      inputs={'dataset': prep_data_run.outputs['cleaned_data']},
                      params={'model_pkg_class': 'sklearn.linear_model.LogisticRegression',
                              'label_column': 'label'})

# test the model using a library (hub://) function and the generated model
test = mlrun.import_function('hub://test_classifier').apply(auto_mount())
test_run = test.run(name="test",
                    params={"label_column": "label"},
                    inputs={"models_path": train_run.outputs['model'],
                            "test_set": train_run.outputs['test_set']})

# push results via notification to Git, Slack, ..
notifier.push_run_results([prep_data_run, train_run, test_run])

# Create model serving function using the new model
serve = mlrun.import_function('hub://v2_model_server').apply(auto_mount())
model_name = 'iris'
serve.add_model(model_name, model_path=train_run.outputs['model'])
addr = serve.deploy()

notifier.push(f"model {model_name} is deployed at {addr}")

# test the model serving function
inputs = [[5.1, 3.5, 1.4, 0.2],
          [7.7, 3.8, 6.7, 2.2]]
my_data = json.dumps({'inputs': inputs})
serve.invoke(f'v2/models/{model_name}/infer', my_data)

notifier.push(f"model {model_name} test passed Ok")
