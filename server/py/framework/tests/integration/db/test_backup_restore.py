# Copyright 2026 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import pathlib
import shutil

import pytest
import sqlalchemy

from mlrun import mlconf

import services.api.utils.db.backup

pytestmark = [
    pytest.mark.skipif(
        os.getenv("MLRUN_TEST_DB", "mysql").lower() != "mysql",
        reason="Backup integration test exercises the mysql code path only",
    ),
    pytest.mark.skipif(
        shutil.which("mysqldump") is None,
        reason="mysqldump CLI is not on PATH",
    ),
    pytest.mark.skipif(
        shutil.which("mysql") is None,
        reason="mysql CLI is not on PATH",
    ),
]


def test_mysql_backup_restore_round_trip_with_complex_password(
    db_engine: sqlalchemy.engine.Engine,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with db_engine.begin() as conn:
        conn.execute(
            sqlalchemy.text(
                "CREATE TABLE backup_marker (id INT PRIMARY KEY, val VARCHAR(64))"
            )
        )
        conn.execute(
            sqlalchemy.text(
                "INSERT INTO backup_marker (id, val) VALUES (1, 'before-backup')"
            )
        )

    monkeypatch.setattr(mlconf.httpdb, "dirpath", str(tmp_path))

    util = services.api.utils.db.backup.DBBackupUtil(backup_rotation=False)

    # If --defaults-extra-file or password escaping is wrong, mysqldump
    # exits with "Access denied" and DBBackupUtil raises RuntimeError.
    util.backup_database("backup.sql")

    backup_path = tmp_path / "mysql" / "backup.sql"
    assert backup_path.exists(), "mysqldump produced no backup file"
    assert backup_path.stat().st_size > 0, "mysqldump produced an empty backup"

    backup_text = backup_path.read_text(errors="ignore")
    assert "CREATE TABLE `backup_marker`" in backup_text
    assert "before-backup" in backup_text

    with db_engine.begin() as conn:
        conn.execute(
            sqlalchemy.text(
                "UPDATE backup_marker SET val = 'after-backup' WHERE id = 1"
            )
        )

    util.load_database_from_backup("backup.sql", "pre_restore.sql")

    with db_engine.connect() as conn:
        restored = conn.execute(
            sqlalchemy.text("SELECT val FROM backup_marker WHERE id = 1")
        ).scalar_one()
    assert restored == "before-backup"
