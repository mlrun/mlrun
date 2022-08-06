# Contributing To MLRun


## Creating a development environment
1. clone your fork and cd into the repo directory
    ```shell script
    git clone git@github.com:<your username>/mlrun.git
    cd mlrun
    ```
2. Set up a virtualenv for running tests (we recommend using venv)
    ```shell script
    python -m venv venv
    source venv/bin/activate
    ```
3. Install mlrun, dependencies and test dependencies
    ```shell script
    make install-requirements
    pip install -e '.[complete]'
    ```
   
## Formatting
We use [black](https://github.com/psf/black) as our formatter, you can basically write your code 
how ever you want, and when you finish black will simply format it for you by running 
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
* System tests - see dedicated section below

## Pull requests
* **Title** - our convention for the pull request title (used as the squashed commit message) is to have it starting with 
[\<scope\>] e.g. "[API] Adding endpoint to list runs"
* **Description** - It's much easier to review when there is a detailed description of the changes, and especially the why-s,
please put effort in writing good description
* **Tests** - we care a lot about tests! if your PR will include good test coverage higher chances it will be merged fast

## System Tests
In the `tests/system/` directory exist test suites to run against a running system, in order to test full MLRun flows.

### Adding System Tests
To add new system tests, all that is required is to create a test suite class which inherits the `TestMLRunSystem`
class from `tests.system.base`. In addition, a special `skip` annotation must be added to the suite, so it won't run 
if the `env.yml` isn't filled. If the test can only run on a full iguazio system and not on an MLRun Kit instance, add
the `enterprise` marker under the `skip` annotation or on the test method itself. If the `enterprise` marker is added
to a specific test method, the `skip` annotation must be added above it in addition to the annotation over the test 
suite. This is because enterprise tests and open source tests require different env vars to be set in the `env.yml`.

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

From here, just use the MLRun sdk within the setup/teardown functions and the tests themselves with regular pytest
functionality. The MLRun SDK will work against the live system you configured, and you can write the tests as you would
any other pytest test.

#### Running System Tests Locally
>**Note** - Running system tests locally is helpful only when adding new tests, and not when testing
> regression on new code changes. For that, see the section below.
1. Ensure you have a running system which is accessible via HTTPS from where you are running the tests.
2. Fill the `tests/system/env.yml` with the `MLRUN_DBPATH`, `V3IO_API`, `V3IO_FRAMESD`, `V3IO_USERNAME` and 
   `V3IO_ACCESS_KEY` (at this moment, `V3IO_PASSWORD` isn't required).
3. Run the system tests by running `make test-system`.

### Checking system test regression on new code
Currently, this can only be done by one of the maintainers, the process is:
1. Push your changes to a branch in the upstream repo
2. Go to the [build action](https://github.com/mlrun/mlrun/actions?query=workflow%3ABuild) and trigger it for the branch 
(leave all options default)
3. Go to the [system test action](https://github.com/mlrun/mlrun/actions?query=workflow%3A%22System+Tests%22) and trigger 
it for the branch, change "Take tested code from action REF" to `true`   

### MLRun API Remote Debugging

Remotely debugging MLRun is possible using PyCharm and `pydevd_pycharm`, 
which is installed on MLRun's api image tagged with "-dev" as its suffix.

To Remote Debug MLRun's API, you may need to:

1. Open PyCharm, [create](https://www.jetbrains.com/help/pycharm/remote-debugging-with-product.html#create-remote-debug-config) a run debug configuration
    - Path mapping `/path/to/mlrun/mlrun=/mlrun`
    - Leave host to be "localhost"
    - Check "Redirect output to console"
    - Uncheck "Suspect after connect"

2. If running remotely - install `ngrok` and run it to create a tunnel. (e.g. `ngrok tcp 40000`)
    - Use the tcp port from (1).

3. Once ngrok is running, tak the public url from the output and use it to connect to the remote debug server.
    ```
    ...
    Forwarding                    tcp://7.tcp.eu.ngrok.io:13209 -> localhost:40000
    ```

4. Add envvars to MLRun's API deployment (or use `mlrun-override-env` configmap) with below envvars
   - `MLRUN_API_DEBUG_MODE` - Whether to enable debugging or not (e.g.: `enabled`)
   - `MLRUN_API_DEBUG_HOST` - The host returned by ngrok (e.g.: `7.tcp.eu.ngrok.io`)
   - `MLRUN_REMOTE_DEBUG_PORT` - The port returned by ngrok (e.g.: `13209`)

5. Start Remote Debugging run configuration created on (1). you will see it is waiting for a connection.

6. Using `kubectl`, patch your mlrun deployment with "dev" image tag.
    ```bash
    kubectl set image deployment/mlrun-api-chief mlrun-api=quay.io/mlrun/mlrun-api:1.0.5-dev
    ```
7. Once the deployment is patched, it will connect to your PyCharm server. Once it is connected, your console
    will log it with `Connected to pydev debugger (build ...)`.

8. You can now use breaking points while running a remote MLRun API instance.
