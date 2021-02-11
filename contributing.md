Contributing To MLRun
=====================

System Tests
------------
In the `tests/system/` directory exist test suites to run against a running system, in order to test full MLRun flows.

### Adding More System Tests
To add more system tests, all that is required is to create a test suite class which inherits the `TestMLRunSystem`
class from `tests.system.base`. In addition, a special `skip` annotation must be added to the suite, so it won't run 
if the `env.yml` isn't filled.

For example:
```python
from tests.system.base import TestMLRunSystem

@TestMLRunSystem.skip_test_if_env_not_configured
class TestSomeFunctionality(TestMLRunSystem):
    def test_the_functionality(self):
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

### Running The System Tests Locally
1. Ensure you have a running system which is accessible via HTTPS from where you are running the tests.
2. Fill the `tests/system/env.yml` with the `MLRUN_DBPATH`, `V3IO_API`, `V3IO_FRAMESD`, `V3IO_USERNAME` and 
   `V3IO_ACCESS_KEY` (at this moment, `V3IO_PASSWORD` isn't required).
3. Run the system tests by running `pytest tests/system`.