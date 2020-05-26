<a id="top"></a>
# Projects

A Project is a container for all your work on a particular activity. All the associated code, jobs and artifacts are organized within the projects. A project is also a great way to collaborate with others, since you can share your work, as well as create projects based on existing projects.
One option is to create a project

It's a best practice to have all your notebooks associated with a project. An easy way to do that is to create a project in the beginning of the notebook using the `new_project` MLRun method, which receives the following parameters:

- **`name`** (Required) &mdash; the project name.
- **`context`** &mdash; the path to a local project directory (the project's context directory).
  The project directory contains a project-configuration file (default: **project.yaml**), which defines the project, and additional generated Python code.
  The project file is created when you save your project (using the `save` MLRun project method), as demonstrated in Step 6.
- **`functions`** &mdash; a list of functions objects or links to function code or objects.
- **`init_git`** &mdash; set to `True` to perform Git initialization of the project directory (`context`).
  > **Note:** It's customary to store project code and definitions in a Git repository.

Projects are visible in the MLRun dashboard only after they're saved to the MLRun database, which happens whenever you run code for a project.

For example, use the following code to create a project named **my-project** and stores the project definition in a subfolder named `conf`:

```python
from os import path
from mlrun import new_project

project_name = 'my-project'
project_path = path.abspath('conf')
project = new_project(project_name, project_path, init_git=True)

print(f'Project path: {project_path}\nProject name: {project_name}')
```

You can also ensure the project name is unique, by concatenating your current username. For example, the following code concatenates the **V3IO_USERNAME** environment variable to the project name:

```python
from os import getenv
project_name = '-'.join(filter(None, ['my-project', getenv('V3IO_USERNAME', None)]))
```

You can use an existing project as a baseline by calling the `load_project` function. This enables reuse of existing code and definitions. For example, to load **my-project** based on a different project stored at `/projects/other_project/conf`:

```python
project = load_project('/projects/other_project/conf', name='my_project')
```

[Back to top](#top)
