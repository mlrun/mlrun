(work-with-git)=
# Using Git to manage projects

You can update the code using the standard Git process (commit, push). If you update/edit the project object you 
need to run `project.save()`, which updates the `project.yaml` file in your context directory, followed by pushing your updates.

You can use the standard `git` cli to `pull`, `commit`, `push`, etc. MLRun project syncs with the local git state.
You can also use project methods with the same functionality. It simplifies the work for common task but does not expose the full git functionality.

* **{py:meth}`~mlrun.projects.MlrunProject.pull`** &mdash; pull/update sources from git or tar into the context dir
* **{py:meth}`~mlrun.projects.MlrunProject.create_remote`** &mdash; create remote for the project git
* **{py:meth}`~mlrun.projects.MlrunProject.push`** &mdash; save project state and commit/push updates to remote git repo

For example: `proj.push(branch, commit_message, add=[])` saves the state to DB & yaml, commits updates, push

```{admonition} Note
You must push updates before you build functions or run workflows which use code from git,
since the builder or containers pull the code from the git repo.
```

If you are using containerized Jupyter you might need to first set your Git parameters, e.g. using the following commands:

```
git config --global user.email "<my@email.com>"
git config --global user.name "<name>"
git config --global credential.helper store
```

After that you need to login once to git with your password, as well as restart the notebook.

``` python
project.push('master', 'some edits')
```