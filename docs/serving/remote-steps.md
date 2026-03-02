(remote-steps)=
# Remote steps

Use RemoteStep in both sync and async engines to invoke an external URI (HTTP or MLRun function). 

## Description

See the full parameter list in {py:class}`~mlrun.serving.remote.RemoteStep`.

## Use cases

## Examples
Using the `async` engine to trigger an external heavy process, such as a service generating a test using model and storing it in a DB would look similar to:
```
flow = function.set_topology("flow", engine="async")
flow.to(name="step1", handler="func1").to(RemoteStep(name="remote_echo", url="https://myservice/path", method="POST")).to(name="laststep", handler="func2").respond()
```

A typical example using the `sync` engine would be to get a prediction from a model:
```
flow = function.get_model_prediction("flow", engine="sync")
flow.to(name="step1", handler="func1").to(RemoteStep(name="prediction", url="http://someservice/path")).to(name="laststep", handler="func2").respond()
```

Example pipeline using an MLRun function URI, for example to get inference from a model:
```
flow = function.set_topology("flow", engine="async")
flow.to(name="step1", handler="func1")
    .to(RemoteStep(name="prediction", url=[project/]name))
    .to(name="laststep", handler="func2").respond()
```
