# Contributing To MLRun

## Creating a development environment

If you are working with an ARM64 machine, please see  [Developing with ARM64 machines](#developing-with-arm64-machines).

We recommend using [pyenv](https://github.com/pyenv/pyenv#installation) to manage your python versions.
Once you have pyenv installed, you can create a new environment by running:

```bash
pyenv install 3.9
```

To activate the environment, run:

```bash
pyenv shell 3.9
```

Or, set as default by running:

```bash
pyenv global 3.9
```


Fork, clone and cd into the MLRun repository directory
```shell script
git clone git@github.com:<your username>/mlrun.git
cd mlrun
```

Set up a virtualenv (we recommend using [venv](https://docs.python.org/3.9/library/venv.html))
```shell script
python -m venv venv
source venv/bin/activate
```

Install MLRun, dependencies and dev dependencies
```shell script
make install-requirements
pip install -e '.[complete]'
```

## Developing with ARM64 machines

Some mlrun dependencies are not yet available for ARM64 machines via pypi, so we need to work with conda to get the packages compiled for ARM64 platform.   
Install Anaconda from [here](https://docs.anaconda.com/free/anaconda/install/index.html) and then follow the steps below:

Fork, clone and cd into the MLRun repository directory
```shell script
git clone git@github.com:<your username>/mlrun.git
cd mlrun
```

Create a conda environment and activate it
```shell script
conda create -n mlrun python=3.9
conda activate mlrun
``` 

Then, install the dependencies
```shell script
make install-conda-requirements
```

Run some unit tests to make sure everything works:
```shell script
python -m pytest ./tests/projects
```

If you encounter any error with 'charset_normalizer' for example:
```shell script
AttributeError: partially initialized module 'charset_normalizer' has no attribute 'md__mypyc' (most likely due to a circular import)
```
Run:
```shell script
pip install --force-reinstall charset-normalizer
```
Finally, install mlrun
```shell script
pip install -e '.[complete]'
```

## Formatting and Linting

We use [ruff](https://docs.astral.sh/ruff/) as our formatter and linter.
Format your code prior opening PR by running:
```shell script
make fmt
```

## Testing

* Lint
    ```shell script
    make lint
    ```

* Unit tests
    ```shell script
    make test-dockerized
    ```

* Integration tests
    ```shell script
    make test-integration-dockerized
    ```

* System tests - see dedicated section below

## Pull requests

* **Title**
  - Begin the title of the PR with `[<scope>]` , with the first letter of the component name in uppercase, e.g `[API] Add endpoint to list runs`.
  - If the PR is addressing a bug, include the keywords `fix` or `bug` in the title of the PR, so that it will be added to the `Bugs & Fixes` section in the release notes.
  - Use imperative verbs when describing the changes made in the PR. For example, instead of writing `Adding endpoint to list runs`, write `Add endpoint to list runs`.
  - Start with a verb after the `[<scope>]` prefix, e.g. `[API] Add endpoint to list runs`.

* **Description** - It's much easier to review when there is a detailed description of the changes, and especially the why-s,
please put effort in writing good description
* **Tests** - we care a lot about tests! if your PR will include good test coverage higher chances it will be merged fast

## System Tests

As we support additional enterprise features while running MLRun in an Iguazio system, some system tests can only run 
on an Iguazio system. To support this, we have two types of system tests.
Using `@pytest.mark.enterprise` markers, we can distinguish between tests that can run on a MLRun Community Edition 
instance and tests that requires and can only run on a full Iguazio system.
Any system test which isn't marked with the `@pytest.mark.enterprise` marker can run on MLRun Community Edition which
incidentally can also be installed locally on a developer machine.

In the `tests/system/` directory exist test suites to run against a running system, in order to test full MLRun flows.

### Setting Up an MLRun Community Edition Instance for System Tests

You can follow the [Install MLRun on Kubernetes](https://docs.mlrun.org/en/latest/install/kubernetes.html) guide to 
install an instance of MLRun Community Edition on your local machine. Notice the mentioned prerequisites and make sure 
you have some kubernetes cluster running locally. 
You can use [minikube](https://minikube.sigs.k8s.io/docs/start/) for this purpose (however this will require an extra step, see below).

### Setting up Test Environment

To run the system tests, you need to set up the `tests/system/env.yml` file. For running the open source system tests,
the only requirement is to set the `MLRUN_DBPATH` environment variable to the url of the mlrun api service which is installed
as part of the MLRun Community Edition installation.
Once the installation is completed, it outputs a list of urls which you can use to access the various services. Similar to:

```
NOTES:
You're up and running !

1. Jupyter UI is available at:
  http://127.0.0.1:30040
...
4. MLRun API is exposed externally at:
  http://127.0.0.1:30070
...
Happy MLOPSing !!! :]
```

Notice the "MLRun API is exposed externally at: http://127.0.0.1:30070" line. This is the url you need to set in the 
`env.yml` file, as the `MLRUN_DBPATH` value..

If running via minikube, you will first need to run
```shell
minikube -n mlrun service mlrun-api
```
Which will tunnel the mlrun api service to your local machine. You can then use the url that is outputted by this command
to set the `MLRUN_DBPATH` environment variable.

### Adding System Tests

To add new system tests, all that is required is to create a test suite class which inherits the `TestMLRunSystem`
class from `tests.system.base`. In addition, a special `skip` annotation must be added to the suite, so it won't run 
if the `env.yml` isn't filled. If the test can only run on a full Iguazio system and not on an [MLRun CE](https://github.com/mlrun/ce) 
instance, add the `enterprise` marker under the `skip` annotation or on the test method itself.
If the `enterprise` marker is added to a specific test method, the `skip` annotation must be added above it in addition to the annotation 
over the test suite.
This is because enterprise tests and open source tests require different env vars to be set in the `env.yml`.

For example:
```python
import pytest
from tests.system.base import TestMLRunSystem

@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestSomeFunctionality(TestMLRunSystem):
    def test_the_functionality(self):
        pass
```

Example of a suite with two tests, one of them meant for enterprise only
```python
import pytest
from tests.system.base import TestMLRunSystem

@TestMLRunSystem.skip_test_if_env_not_configured
class TestSomeFunctionality(TestMLRunSystem):

    def test_open_source_features(self):
        pass

    @TestMLRunSystem.skip_test_if_env_not_configured
    @pytest.mark.enterprise
    def test_enterprise_features(self):
        pass
```

If some setup or teardown is required for the tests in the suite, add these following functions to the suite:
```python
from tests.system.base import TestMLRunSystem

@TestMLRunSystem.skip_test_if_env_not_configured
class TestSomeFunctionality(TestMLRunSystem):
    
    def custom_setup(self):
        pass
    
    def custom_teardown(self):
        pass
    
    def test_the_functionality(self):
        pass
```

From here, just use the MLRun SDK within the setup/teardown functions and the tests themselves with regular pytest
functionality. The MLRun SDK will work against the live system you configured, and you can write the tests as you would
any other pytest test.

### You're Done!

All that's left now is to run whichever open source system tests you want to run. You can run them all by running the 
command
```shell
make test-system-open-source
```

### Checking system test regression on new code

Currently, this can only be done by one of the maintainers, the process is:
1. Push your changes to a branch in the upstream repo
2. Go to the [build action](https://github.com/mlrun/mlrun/actions?query=workflow%3ABuild) and trigger it for the branch 
(leave all options default)
3. Go to the [system test action](https://github.com/mlrun/mlrun/actions?query=workflow%3A%22System+Tests%22) and trigger 
it for the branch, change "Take tested code from action REF" to `true`   

## Migrating to Python 3.9

MLRun moved to Python 3.9 from 1.3.0.  
If you are working on MLRun 1.2.x or earlier, you will need to switch between python 3.9 and python 3.7 interpreters.
To work with multiple python interpreters, we recommend using _pyenv_ (see [Creating a development environment](#creating-a-development-environment)).
Once you have pyenv installed, create multiple `venv` for each Python version, so when you switch between them, you will
have the correct dependencies installed. You can manage and switch venvs through PyCharm project settings.

e.g.:

```bash
pyenv shell 3.9
pyenv virtualenv mlrun

pyenv shell 3.7
pyenv virtualenv mlrun37
```
