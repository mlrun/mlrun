import pytest

import mlrun


def test_nopdb():
    # by default we use a nopdb with raise_error = False
    assert mlrun.mlconf.httpdb.nop_db.raise_error is False

    rundb = mlrun.get_run_db()
    assert isinstance(rundb, mlrun.db.NopDB)

    # not expected to fail as it in the white list
    rundb.connect()

    # not expected to fail
    rundb.read_run("123")

    # set raise_error to True
    mlrun.mlconf.httpdb.nop_db.raise_error = True

    assert mlrun.mlconf.httpdb.nop_db.raise_error is True

    # not expected to fail as it in the white list
    rundb.connect()

    # expected to fail
    with pytest.raises(mlrun.errors.MLRunBadRequestError):
        rundb.read_run("123")
