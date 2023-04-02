from mlrun.runtimes.local import run_exec


def test_run_exec_basic():
    out, err = run_exec(["echo"], ["hello"])
    assert out == "hello\n"
    assert err == ""


# ML-3710
def test_run_exec_verbose_stderr():
    out, err = run_exec(["python"], ["assets/verbose_stderr.py"])
    assert out == "some output\n"
    assert len(err) == 100000
