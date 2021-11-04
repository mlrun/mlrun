
## Github/Gitlab and CI/CD Integration


Users may want to run their ML Pipelines using CI frameworks like Github Actions, GitLab CI/CD, etc.
MLRun support simple and native integration with the CI systems, see the following example in which we combine 
local code (from the repository) with MLRun marketplace functions to build an automated ML pipeline which:

* runs data preparation
* train a model
* test the trained model
* deploy the model into a cluster
* test the deployed model

