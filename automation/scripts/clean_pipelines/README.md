# Clean Pipelines Script 

This script is designed to delete old pipeline runs associated with a specific MLRun project,
with flexible date filtering.
You can use this as an MLRun function to automate pipeline cleanup tasks.

## Usage

### Step 1: Set up the project

First, create or get an existing MLRun project:

```python
import mlrun

project_name = "your_project_name"
project = mlrun.get_or_create_project(project_name, context="./")
```

### Step 2: Set up the function and parameters

Add the `clean_pipelines.py` script to your project at the location of your choice. Set up the function as follows:

```python
code_path = "/path/to/script/clean_pipelines.py"
function_name = "function-name"

func = project.set_function(
    name=function_name,
    func=code_path,
    kind="job",
    image="mlrun/mlrun",
    handler="delete_project_old_pipelines",
)
```

The `delete_project_old_pipelines` function deletes old pipeline runs associated with a specified project.

#### Function parameters:
- **context**: The context object to log results (automatically passed by MLRun).
- **project_name** (str): The name of the project for which to delete old pipelines.
- **end_date** (str): The cutoff date for deleting pipeline runs. All runs created on or before this date will be considered for deletion (format: YYYY-MM-DDTHH:MM:SSZ).
- **start_date** (str, optional): If provided, only runs created on or after this date will be considered for deletion (default is empty, no filtering).
- **dry_run** (bool): If True, only log what would be deleted (default: False).

### Define the `end_date` and any other necessary parameters, for example:

```python
end_date = "2024-10-20T10:10:06Z"
params = {
    "project_name": "remote-workflow-example1",
    "end_date": end_date,
    "dry_run": False,
}
```

### Step 3: Build the function

Use MLRun version ***1.7.0*** or greater to build the function as follows:

```python
function_build = project.build_function(
    func,
    base_image="python:3.9",
    with_mlrun=False,
    force_build=True,
    requirements=["mlrun==1.7.0-rc55"],
)
```

### Step 4: Run the function using the defined parameters:
```python
job = function_build.function.run(params=params)
```

### Step 5: Check the Job Outputs

After running the function, you can check the outputs of the job to verify the results. Use the following code to access the job outputs:

```python
job.outputs
```

