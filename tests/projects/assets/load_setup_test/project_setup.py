import mlrun


def setup(project: mlrun.projects.MlrunProject):
    """Example for project setup script which modify project metadata and functions"""
    project.spec.params["test123"] = "456"
    prep_func = project.set_function(
        "prep_data.py", "prep-data", kind="job", image="mlrun/mlrun"
    )
    prep_func.set_label("tst1", project.get_param("p2"))

    srv_func = project.set_function(
        "serving.py", "serving", kind="serving", image="mlrun/mlrun"
    )
    # graph = srv_func.set_topology()
    srv_func.add_model("x", ".", class_name="MyCls")
    return project
