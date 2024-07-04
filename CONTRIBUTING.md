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

Using [UV](https://github.com/astral-sh/uv) is also an option, 
you may need to override current MLRun default packager-installer env var `MLRUN_PYTHON_PACKAGE_INSTALLER` to `uv`

```shell script
uv venv venv --seed
source venv/bin/activate
```

Install MLRun, dependencies and dev dependencies
```shell script
make install-requirements
pip install -e '.[complete]'
```

## Developing with ARM64 machines

Some MLRun dependencies are not yet available for ARM64 machines via pypi, 
so we need to work with conda to get the packages compiled for ARM64 platform.   

Fork, clone and cd into the MLRun repository directory
```shell script
git clone git@github.com:<your username>/mlrun.git
cd mlrun
```

Create a [Conda](https://docs.anaconda.com/free/anaconda/install/index.html) environment and activate it
```shell script
conda create -n mlrun python=3.9
conda activate mlrun
```

Then, install the dependencies
```shell script
make install-conda-requirements
```

*Or*, alternatively, you may use native Python atop the ARM64 machine, but you will need to compile some dependencies.
Execute below script to overcome MLRun current protobuf~3.20 incompatibility.

NOTE: This script will compile and install protobuf 3.20.0 for macOS arm64. At the end of the installation, it will
ask you to type your password to finish installing the compiled protobuf.

```shell script
./automation/scripts/compile_protobuf320_for_mac_arm64.sh
make install-requirements
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
  - If the PR is addressing a bug, include the keywords `fix` or `bug` in the title of the PR, so that it will be added
  to the `Bugs & Fixes` section in the release notes.
  Additionally, the PR title should reflect how the fix was done and not how it was fixed.
  For example if there was a race condition where an artifact got deleted and created at the same time, instead of
  writing "Fixed artifact locking mechanism", write "Fixed artifact creation/deletion race condition".
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

### Python Code Conventions:

1. Use snake_case for parameters, functions and module names.
2. Use CamelCase for Class names.
3. Style conventions (handled by linter) can be found here: https://pycodestyle.pycqa.org/en/latest/intro.html#error-codes
4. Functions must be kept short, read like a story and break into smaller functions when needed.
5. Use meaningful names and avoid abbreviations for variables, functions and modules to enhance readability.
6. Avoid using hardcoded values, use Enum values, env variables or configuration files instead.
7. Use typing hints for complex data structures.
8. Use tmp_path fixture for temporary files/directories in tests.
9. When using a function in the same file, it should be declared below the calling function as it makes it easier to understand the functionality.
10. Private methods should be declared below the public ones.
11. When dealing with numerical values representing sizes or time related variables, add a comment to specify the unit
(e.g., KB, MB, seconds, hours) for clarity, or alternatively, use variable names like “size_in_KB” for explicit unit indication.
12. After updating an existing code, ensure that the old documentation is synced with the newly added code or needs
to be updated accordingly.
13. When importing from local code, NEVER use: `from X import Y` instead use: `import X` or `import X as Y`.
For external packages, it's acceptable to use `from X import Y`  since they won't try to import back to our code.
14. Docstring format, for all public API-level functions:
Format: Use triple quotes (""" """) for docstrings.
Description: Provide a brief and informative description of the function's purpose.
Parameters: List all parameters with their data types and a brief description. Use the :param tag for each parameter.
Return Value: If the function returns a value, describe it using the :return tag.
Example:

```
def function_name(parameter1, parameter2):
	"""
	Brief description of the function's purpose.
	:param parameter1: Description of parameter1.
	:param parameter2: Description of parameter2.
	:return: Description of the return value (if applicable).
	"""
	# Function implementation
```

15. When calling functions with multiple parameters, prefer using keyword arguments to improve readability and clarity.
16. Logging: use structured variable instead of f-strings, for example: `logger.debug("Message", var1=var1, ...)`, and
try to avoid logging large objects which are hard to decipher.
17. Use f-strings for string formatting instead of the old `.format(...)` except when dealing with template strings.

### MLRun Coding Conventions:

1. When converting an error object to a string representation, instead of using: `str(error)` use: `mlrun.errors.err_to_str(error)`
2. Use `mlrun.mlconf` Instead of `mlrun.config.config`.
3. When deprecating a parameter/method/class we keep backwards compatibility for 2 minor versions.
For example if we deprecated a parameter in 1.6.0, it will be removed in 1.8.0.
Always specify what should be used instead. If there is nothing to be used instead, specify why.

* Deprecating a parameter:
Check if the parameter is given and output a FutureWarning and add a TODO with when this should be removed to
help developers keep track.
for example:

```
if uid:
	warnings.warn(
		"'uid' is deprecated in 1.6.0 and will be removed in 1.8.0, use 'tree' instead.",
		# TODO: Remove this in 1.8.0
		FutureWarning,
	)
```

* Deprecating a method:
Use 'deprecated'

```
# TODO: remove in 1.6.0
@deprecated(
	version="1.4.0",
	reason="'verify_base_image' will be removed in 1.6.0, use 'prepare_image_for_deploy' instead",
	category=FutureWarning,
)
def verify_base_image(self):
```

* Deprecating a class:

```
# TODO: Remove in 1.7.0
@deprecated(
	version="1.5.0",
	reason="v1alpha1 mpi will be removed in 1.7.0, use v1 instead",
	category=FutureWarning,
)
class MpiRuntimeV1Alpha1(AbstractMPIJobRuntime):
```
4. Minimize imports and avoid unnecessary dependencies in client code.
5. Scale performance: be caution when executing large queries in order to prevent overloading the database.
