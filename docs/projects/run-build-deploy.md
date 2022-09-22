(run_project_functions)=
# Run, build, and deploy functions

There is a set of methods used to deploy and run project functions. They can be used interactively or inside a pipeline. 
When used inside a pipeline, the methods are automatically mapped to the relevant pipeline engine command.

* {py:meth}`~mlrun.projects.run_function`  - Run a local or remote task as part of a local run or pipeline
* {py:meth}`~mlrun.projects.build_function`  - deploy an ML function, build a container with its dependencies for use in runs
* {py:meth}`~mlrun.projects.deploy_function`  - deploy real-time/online (nuclio or serving based) functions

You can use those methods as `project` methods, or as global (`mlrun.`) methods. The current project is assumed for the later case.

    run = myproject.run_function("train", inputs={"data": data_url})  # will run the "train" function in myproject
    run = mlrun.run_function("train", inputs={"data": data_url})  # will run the "train" function in the current/active project
    
The first parameter in those three methods is the function name (in the project), or it can be a function object if you want to 
use functions that you imported/created ad hoc, for example:

    # import a serving function from the marketplace and deploy a trained model over it
    serving = import_function("hub://v2_model_server", new_name="serving")
    deploy = deploy_function(
        serving,
        models=[{"key": "mymodel", "model_path": train.outputs["model"]}],
    )
