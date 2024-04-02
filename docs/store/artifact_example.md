(data-items)=
# Artifact Types

MLRun support logging a couple of artifact types in this page you will see examples of how to log different artifacts in MLRun.
```{admonition} Artifact Types Supported by MLRun
    mlrun.artifacts.Artifact - Artifcat
    mlrun.artifacts.DirArtifact - Directory
    mlrun.artifacts.LinkArtifact - Link
    mlrun.artifacts.PlotArtifact - Plot
    mlrun.artifacts.ChartArtifact - Chart
    mlrun.artifacts.TableArtifact - Table
    mlrun.artifacts.PlotlyArtifact - Plotly
    mlrun.artifacts.BokehArtifact - Bokeh
```
for logging artifacts types we will use the [log_artifact](https://docs.mlrun.org/en/latest/api/mlrun.execution.html#mlrun.execution.MLClientCtx.log_artifact) method, this method can be used by the Project object or the context object if using in a run.
```{admonition} Note
This Page will cover example for all artifcats types execpt Models and Datasets. this is becuase for model and dataset you can find example in the See also section.
```
**In this section**
- [Example logging an artifact](#logging-artifact)
- [Example logging a directory artifact](#logging-directory-artifact)
- [Example logging a Plotly artifact](#logging-plotly-artifact)


**See also:**
- {ref}`working-with-data-and-model-artifacts`
- {ref}`models`
- {ref}`logging_datasets`
- [Logging a Databricks response as an artifact](../runtimes/databricks.html#logging-a-databricks-response-as-an-artifact).

### Logging Artifact
***Note -*** when using `log_artifact` and not specifying the artifact object to a file by default the artifact type is 
`mlrun.artifacts.Artifact`
```python
    project.log_artifact(
        "some-data",
        body=b"abc is 123",
        local_path="file.txt",
        labels={"Test": "label-test"},
    )
```
In addition, `log_artifact` can use to log many kind of files for example `html`,`pkl`.
for log a file that is not a `.txt` file you can simply point to the file from you local_path or use the body as the artifact content by using the `body` flag.
for example how to log a plotly figure using `log_artifact` to a `html` file:
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

### Logging an Artifact Directory
***Note -*** when using `log_artifact` and not specifying the artifact object and pointing to a directory by default the artifact type is 
`mlrun.artifacts.DirArtifact`
```python
    project.log_artifact(
    "artifact-directory-testing",
    local_path="./artifact_directory/",
    labels={"Dir":"dir-example"})
```
### Logging a Link Artifact
....

### Logging a Plot Artifact


