(data-items)=
# Logging Artifacts

For logging artifacts that are not Models or Datasets we will use the [log_artifact](https://docs.mlrun.org/en/latest/api/mlrun.execution.html#mlrun.execution.MLClientCtx.log_artifact) method.

This method can be used by the Project object or the context object when logging an artifact in runtime .e.g job.

**In this section**
- [Examples of logging an artifact](#logging-artifact)
- [Example of logging a directory artifact](#logging-directory-artifact)

**See also:**
- {ref}`working-with-data-and-model-artifacts`
- {ref}`models`
- {ref}`logging_datasets`
- [Logging a Databricks response as an artifact](../runtimes/databricks.html#logging-a-databricks-response-as-an-artifact).

### Logging Artifact
***Simple example***
```python
    project.log_artifact(
        "some-data",
        body=b"abc is 123",
        local_path="file.txt",
        labels={"Test": "label-test"},
    )
```
In addition, `log_artifact` can be used to log many types of files for example `html`,`pkl` and more.

For log an artifacts file you can simply use the local file path to the file using the `local_path` flag or use the `body` flag to dumps the object content.
you can see below an example how to log a plotly figure using `log_artifact` to a `html` file:

***Log plotly object as a html file***
```python
import plotly.graph_objects as go
import numpy as np


# Create a Sin(x) Graph
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a Plotly figure
fig = go.Figure()

# Add a line trace to the figure
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Sin(x)'))

# Update layout
fig.update_layout(
    title="Sin(x) Plot",
    xaxis_title="x",
    yaxis_title="sin(x)",
    template="plotly_dark" 
)

project.log_artifact("plotly-art",
                     body=fig.to_html(),
                     format="html")
```

```{admonition} Note
Please note that for every object thier is a diffrent way to dumbs to convert the object to be log the artifact using the body flag
```

### Logging an Plotly Artifacts 
Below an example how to easily log a plotly artifact that can be done using `mlrun.artifacts.PlotlyArtifact` object. 
```python
    plotly_artifact = mlrun.artifacts.PlotlyArtifact(figure=fig, key="sin_x")
    context.log_artifact(plotly_artifact)
```
### Logging an Directory Artifact 
when using `log_artifact` and pointing to log a directory by default the artifact type is 
logged an `mlrun.artifacts.DirArtifact` object.
```python
    project.log_artifact(
    "artifact-directory-testing",
    local_path="./artifact_directory/",
    labels={"Dir":"dir-example"})
```



