
## Github/Gitlab and CI/CD Integration

Users may want to run their ML Pipelines using CI frameworks like Github Actions, GitLab CI/CD, etc.
MLRun support simple and native integration with the CI systems, users can implement workflows which we combine 
local code (from the repository) with MLRun marketplace functions to build an automated ML pipeline which:

* runs data preparation
* train a model
* test the trained model
* deploy the model into a cluster
* test the deployed model

MLRun workflows can run inside the CI system, we will ususlly use the `mlrun project` CLI command to load the project 
and run a workflow as part of a code update (e.g. pull request, etc.). The pipeline tasks will be executed on the Kubernetes cluster which is orchestrated by MLRun.

See details:
* [**Using GitHub Actions**](#using-github-actions)
* [**Using GitLab CI/CD**](#using-gitlab-ci-cd)

When MLRun is executed inside a [GitHub Actions](https://docs.github.com/en/actions) or [GitLab CI/CD](https://docs.gitlab.com/ee/ci/) pipeline it will detect the environment attributes automatically 
(e.g. repo, commit id, etc..), in addition few environment variables and credentials must be set.

* **MLRUN_DBPATH** - url of the mlrun cluster
* **V3IO_USERNAME** - username in the remote iguazio cluster
* **V3IO_ACCESS_KEY** - access key to the remote iguazio cluster
* **GIT_TOKEN** or **GITHUB_TOKEN** - Github/Gitlab API Token (will be set automatically in Github Actions)
* **SLACK_WEBHOOK** - optional, Slack API key when using slack notifications

When the workflow runs inside the Git CI system it will report the pipeline progress and results back into the Git tracking system

**The results will appear in the CI system in the following way:**

<img src="../_static/images/git-pipeline.png" alt="mlrun-architecture" width="800"/><br>


### Using GitHub Actions

when running using [GitHub Actions](https://docs.github.com/en/actions) the user need to set the credentials/secrets 
and add a script under the `.github/workflows/` directory which will be executed when code is commited/pushed.

Example script, will be invoked when we add the comment "/run" to our pull request:

```yaml
name: mlrun-project-workflow
on: [issue_comment]

jobs:
  submit-project:
    if: github.event.issue.pull_request != null && startsWith(github.event.comment.body, '/run')
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python 3.7
      uses: actions/setup-python@v1
      with:
        python-version: '3.7'
        architecture: 'x64'
    
    - name: Install mlrun
      run: python -m pip install pip install mlrun
    - name: Submit project
      run: python -m mlrun project ./ -w -r main ${CMD:5}
      env:
        V3IO_USERNAME: ${{ secrets.V3IO_USERNAME }}
        V3IO_API: ${{ secrets.V3IO_API }}
        V3IO_ACCESS_KEY: ${{ secrets.V3IO_ACCESS_KEY }}
        MLRUN_DBPATH: ${{ secrets.MLRUN_DBPATH }}
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }} 
        SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK }}
        CMD: ${{ github.event.comment.body}}
```

See the full example in [**https://github.com/mlrun/project-demo**](https://github.com/mlrun/project-demo)


### Using GitLab CI/CD

when running using [GitLab CI/CD](https://docs.gitlab.com/ee/ci/) the user need to set the credentials/secrets 
and update the script `.gitlab-ci.yml` directory which will be executed when code is commited/pushed.

Example script, will be invoked when we create a pull request (merge requests):

```yaml
image: mlrun/mlrun

run:
  script:
    - python -m mlrun project ./ -w -r ci
  only:
    - merge_requests
```

See the full example in [**https://gitlab.com/yhaviv/test2**](https://gitlab.com/yhaviv/test2)
