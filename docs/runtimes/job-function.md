(job-function)=
# Function of type `job`

You can deploy a model using a `job` type function, which runs the code in a Kubernetes Pod. 

You can create (register) a `job` function with basic attributes such as code, requirements, image, etc. using the 
{py:meth}`~mlrun.projects.MlrunProject.set_function` method.
You can also import an existing job function/template from the {ref}`function-hub`.

Functions can be created from a single code, notebook file, or have access to the entire project context directory. 
(By adding the `with_repo=True` flag, the project context is cloned into the function runtime environment.) 

Examples:


```python
# register a (single) python file as a function
project.set_function('src/data_prep.py', name='data-prep', image='mlrun/mlrun', handler='prep', kind="job")

# register a notebook file as a function, specify custom image and extra requirements 
project.set_function('src/mynb.ipynb', name='test-function', image="my-org/my-image",
                      handler="run_test", requirements=["scikit-learn"], kind="job")

# register a module.handler as a function (requires defining the default sources/work dir, if it's not root)
project.spec.workdir = "src"
project.set_function(name="train", handler="training.train",  image="mlrun/mlrun", kind="job", with_repo=True)
```

To run the job:
```
project.run_function("train")
```

**See also**
- [Create and register functions](../runtimes/create-and-use-functions.html)
- [How to annotate notebooks (to be used as functions)](../runtimes/mlrun_code_annotations.html)
- [How to run, build, or deploy functions](./run-build-deploy.html)
- [Using functions in workflows](./build-run-workflows-pipelines.html)