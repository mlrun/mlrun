def handler(context):
    param1 = context.get_param("param1")
    project_param = context.get_project_param("project_param")
    context.log_result("param1", param1)
    context.log_result("project_param", project_param)
