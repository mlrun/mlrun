import subprocess

acceptable_stderr_errors = [
    "Kubeflow Pipelines (KFP) is not installed. Using noop implementations."
]


def test_import_mlrun():
    out = subprocess.run(["python", "-c", "import mlrun"], capture_output=True)
    stdout_lines = out.stdout.decode("utf-8").strip().split("\n")
    stderr_lines = out.stderr.decode("utf-8").strip().split("\n")
    unexpected_stdout_errors = [line for line in stdout_lines if "[error]" in line]
    unexpected_stderr_errors = [
        line for line in stderr_lines if line not in acceptable_stderr_errors
    ]
    assert unexpected_stdout_errors == [], "`import mlrun` wrote unexpected error logs"
    assert (
        unexpected_stderr_errors == []
    ), "`import mlrun` wrote unexpected errors to stderr"
