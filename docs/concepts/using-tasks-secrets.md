(using-tasks-with-secrets)=
# Using tasks with secrets
MLRun uses the concept of Tasks to encapsulate runtime parameters. Tasks are used to specify execution context
such as hyper-parameters. They can also be used to pass details about secrets that are going to be used in the 
runtime. This allows for control over specific secrets passed to runtimes, and support for the various MLRun secret
providers.

To pass secret parameters, use the Task's {py:func}`~mlrun.model.RunTemplate.with_secrets()` function. For example, 
the following command passes specific project-secrets to the execution context:

```{code-block} python
:emphasize-lines: 8-8

function = mlrun.code_to_function(
    name="secret_func",
    filename="my_code.py",
    handler="test_function",
    kind="job",
    image="mlrun/mlrun"
)
task = mlrun.new_task().with_secrets("kubernetes", ["AWS_KEY", "DB_PASSWORD"])
run = function.run(task, ...)
```

The {py:func}`~mlrun.model.RunTemplate.with_secrets()` function tells MLRun what secrets the executed code needs to 
access. The MLRun framework prepares the needed infrastructure to make these secrets available to the runtime, 
and passes information about them to the execution framework by specifying those secrets in the spec of the runtime. 
For example, if running a kubernetes job, the secret keys are noted in the generated pod's spec.

The actual details of MLRun's handling of the secrets differ per the **secret provider** used. The following sections
provide more details on these providers and how they handle secrets and their values.

Regardless of the type of secret provider used, the executed code uses the 
{py:func}`~mlrun.execution.MLClientCtx.get_secret()` API to gain access to the value of the secrets passed to it, 
as shown in the above example.