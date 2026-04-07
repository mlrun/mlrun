import os

_real_os_exit = os._exit


def _coverage_saving_exit(status):
    """Save coverage data before os._exit() in a forked child."""
    try:
        import coverage

        cov = coverage.Coverage.current()
        if cov is not None:
            cov.stop()
            cov.save()
    except Exception:
        pass
    _real_os_exit(status)


def _patch_exit_for_coverage():
    os._exit = _coverage_saving_exit


if os.environ.get("COVERAGE_PROCESS_START"):
    os.register_at_fork(after_in_child=_patch_exit_for_coverage)
