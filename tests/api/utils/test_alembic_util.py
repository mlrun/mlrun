import os.path
import pathlib
import shutil
import typing
import unittest.mock

import alembic
import alembic.config
import pytest

import mlrun.api.utils.db.alembic
from mlrun import mlconf


class Constants(object):
    revision_history = ["revision2", "revision1"]
    initial_revision = "revision1"
    latest_revision = "revision2"
    unknown_revision = "revision3"


@pytest.mark.parametrize("from_scratch", [True, False])
def test_no_database_exists(
    mock_alembic, mock_database, mock_shutil_copy, from_scratch
):
    mock_database(db_file_exists=False)
    alembic_util = mlrun.api.utils.db.alembic.AlembicUtil(pathlib.Path(""))
    alembic_util.init_alembic(from_scratch=from_scratch)
    assert mock_alembic.stamp_calls == []
    assert mock_alembic.upgrade_calls == ["head"]
    mock_shutil_copy.assert_not_called()


@pytest.mark.parametrize("from_scratch", [True, False])
def test_database_exists_no_revision(
    mock_alembic, mock_database, mock_shutil_copy, from_scratch
):
    mock_database()
    alembic_util = mlrun.api.utils.db.alembic.AlembicUtil(pathlib.Path(""))
    alembic_util.init_alembic(from_scratch=from_scratch)

    # from scratch should skip stamp even if no revision exists
    expected_stamp_calls = ["revision1"] if not from_scratch else []
    assert mock_alembic.stamp_calls == expected_stamp_calls
    assert mock_alembic.upgrade_calls == ["head"]
    mock_shutil_copy.assert_not_called()


@pytest.mark.parametrize("from_scratch", [True, False])
def test_database_exists_known_revision(
    mock_alembic, mock_database, mock_shutil_copy, mock_db_file_name, from_scratch
):
    mock_database(current_revision=Constants.initial_revision)
    alembic_util = mlrun.api.utils.db.alembic.AlembicUtil(pathlib.Path(""))
    alembic_util.init_alembic(from_scratch=from_scratch)
    assert mock_alembic.stamp_calls == []
    assert mock_alembic.upgrade_calls == ["head"]
    mock_shutil_copy.assert_called_once_with(
        mock_db_file_name, pathlib.Path(f"{Constants.initial_revision}.db")
    )


@pytest.mark.parametrize("from_scratch", [True, False])
def test_database_exists_unknown_revision_successful_downgrade(
    mock_alembic, mock_database, mock_shutil_copy, mock_db_file_name, from_scratch
):
    mock_database(current_revision=Constants.unknown_revision)
    alembic_util = mlrun.api.utils.db.alembic.AlembicUtil(pathlib.Path(""))
    alembic_util.init_alembic(from_scratch=from_scratch)
    assert mock_alembic.stamp_calls == []
    assert mock_alembic.upgrade_calls == ["head"]
    copy_calls = [
        # first copy - backup the current database before downgrading
        unittest.mock.call(
            mock_db_file_name, pathlib.Path(f"{Constants.unknown_revision}.db")
        ),
        # second copy - to downgrade to the old db file
        unittest.mock.call(
            pathlib.Path(f"{Constants.latest_revision}.db"), mock_db_file_name
        ),
        # third copy - to back up the db file. In a real scenario the backup would be {latest_revision}.db
        # as the revision should change during the last copy, but changing a mock during the init_alembic function
        # is cumbersome and might make the test unreadable - so the current revision stays unknown_revision.
        unittest.mock.call(
            mock_db_file_name, pathlib.Path(f"{Constants.unknown_revision}.db")
        ),
    ]
    mock_shutil_copy.assert_has_calls(copy_calls)


@pytest.mark.parametrize("from_scratch", [True, False])
def test_database_exists_unknown_revision_failed_downgrade(
    mock_alembic, mock_database, mock_shutil_copy, mock_db_file_name, from_scratch
):
    mock_database(
        current_revision=Constants.unknown_revision, db_backup_exists=False,
    )
    alembic_util = mlrun.api.utils.db.alembic.AlembicUtil(pathlib.Path(""))
    with pytest.raises(
        RuntimeError,
        match=f"Cannot fall back to revision {Constants.latest_revision}, "
        f"no back up exists. Current revision: {Constants.unknown_revision}",
    ):
        alembic_util.init_alembic(from_scratch=from_scratch)

    assert mock_alembic.stamp_calls == []
    assert mock_alembic.upgrade_calls == []
    mock_shutil_copy.assert_not_called()


@pytest.fixture()
def mock_database(
    monkeypatch, mock_alembic, mock_db_file_name
) -> typing.Callable[[typing.List[str], str, bool, bool], None]:
    def _mock_database(
        revision_history: typing.List[str] = None,
        current_revision: str = "",
        db_file_exists: bool = True,
        db_backup_exists: bool = True,
    ):
        revision_history = revision_history or Constants.revision_history

        def _db_file_exists(file_name: str) -> bool:
            if file_name == mock_db_file_name:
                return db_file_exists
            else:
                return db_backup_exists

        monkeypatch.setattr(os.path, "isfile", _db_file_exists)

        def _current_revision(alembic_config: typing.Any):
            if current_revision != "" and current_revision not in revision_history:
                raise Exception(
                    f"Can't locate revision identified by '{current_revision}'"
                )

            alembic_config.print_stdout(current_revision)

        mock_alembic.current = _current_revision

        def _revision_history(alembic_config: typing.Any):
            for revision in revision_history:
                alembic_config.print_stdout(f"none -> {revision}, revision name")

        mock_alembic.history = _revision_history

    return _mock_database


@pytest.fixture()
def mock_db_file_name(monkeypatch) -> str:
    db_file_name = "test.db"
    monkeypatch.setattr(mlconf.httpdb, "dsn", db_file_name)
    return db_file_name


@pytest.fixture()
def mock_shutil_copy(monkeypatch) -> unittest.mock.Mock:
    copy = unittest.mock.Mock()
    monkeypatch.setattr(shutil, "copy2", copy)
    return copy


class MockAlembicCommand(object):
    def __init__(self):
        self.stamp_calls = []
        self.upgrade_calls = []

    def stamp(self, alembic_config: typing.Any, revision: str):
        self.stamp_calls.append(revision)

    def upgrade(self, alembic_config: typing.Any, revision: str):
        self.upgrade_calls.append(revision)


@pytest.fixture()
def mock_alembic(monkeypatch) -> MockAlembicCommand:
    mocked_alembic_command = MockAlembicCommand()
    monkeypatch.setattr(alembic, "command", mocked_alembic_command)
    monkeypatch.setattr(alembic.config, "Config", unittest.mock.Mock())
    return mocked_alembic_command
