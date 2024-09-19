(log-artifacts)=
# Logging artifacts
To log artifacts that are not of kind model or dataset, use the {py:meth}`~mlrun.execution.MLClientCtx.log_artifact` method.
You can apply this method to the project object or the context object when logging an artifact in runtime, for example a job.

**In this section**
- [Basic logging of an artifacts file](#basic-logging-of-an-artifacts-file)
- [Log a Plotly object as an HTML file](#log-a-plotly-object-as-an-html-file)
- [Logging Plotly artifacts](#logging-plotly-artifacts)
- [Logging directory artifacts](#logging-directory-artifacts)

**See also**
- {ref}`working-with-data-and-model-artifacts`
- {ref}`models`
- {ref}`logging_datasets`
- [Logging a Databricks response as an artifact](../runtimes/databricks.html#logging-a-databricks-response-as-an-artifact)
## Basic logging of an artifacts file
`log_artifact` can be used to log many kinds of files, for example `html`,`pkl` and more. This is the most general method of logging artifacts. 
```{admonition} Tip
Each object type requires a different way to convert it for logging the object to a file. This is just an example of one type.
```
To log an artifacts file, specify the local file path to the file using the `local_path`, or use the `body` to dump the object content.

**Log with local path**
```python
with open("file.txt", "w") as f:
    f.write("abc is 123")

project.log_artifact(
    "file-example",
    local_path="file.txt",
    labels={"Test": "label-test"},
)
```

**Log with body**
```python
project.log_artifact(
    "some-data",
    body=b"abc is 123",
    format="txt",
    labels={"Test": "label-test"},
)
```

## Log a Plotly object as an HTML file
This example illustrates logging a Plotly figure using `log_artifact` as an `html` file:
```python
import plotly.graph_objects as go
import numpy as np

# Create a Sin(x) Graph
x = np.linspace(0, 10, 100)
y = np.sin(x)
# Create a Plotly figure
fig = go.Figure()
# Add a line trace to the figure
fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Sin(x)"))
# Update layout
fig.update_layout(
    title="Sin(x) Plot", xaxis_title="x", yaxis_title="sin(x)", template="plotly_dark"
)
project.log_artifact(
    "plotly-artifact",
    body=fig.to_html(),  # convert object for logging an html file
    format="html",
)
```
## Logging Plotly artifacts 
This example illustrates using MLRun to convert and handle the object: 
```python
# Use mlrun to convert the python object
plotly_artifact = mlrun.artifacts.PlotlyArtifact(figure=fig, key="sin_x")
# Log the artifact
context.log_artifact(plotly_artifact)
```
## Logging directory artifacts 
When using `log_artifact` to log a directory, by default:
- The artifact is logged as an `mlrun.artifacts.DirArtifact` object.
- The files are not uploaded. If you want to upload the files, set `upload=True`.
```python
project.log_artifact(
    "artifact-directory-testing",
    local_path="./artifact_directory/",
    labels={"Dir": "dir-example"},
)
```