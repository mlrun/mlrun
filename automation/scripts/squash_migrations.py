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

"""Squash alembic migrations from root up to and including a target revision
into a single new migration file with the same revision ID.

Usage:
    python squash_migrations.py <target_revision>

Must be run from server/py/services/api/ (where alembic.ini lives).
PYTHONPATH and MLRUN_HTTPDB__DSN must be set (done by squash_migrations_mysql.sh).
"""

import pathlib
import sys
from collections.abc import Iterable, Iterator

import sqlalchemy
import sqlalchemy.pool
from alembic import command
from alembic.autogenerate import produce_migrations, render_python_code
from alembic.config import Config
from alembic.operations import MigrateOperation
from alembic.operations.ops import DowngradeOps, DropIndexOp, ModifyTableOps
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory

ALEMBIC_INI = "alembic.ini"
VERSIONS_DIR = pathlib.Path("migrations/versions")


def _load_config() -> Config:
    return Config(ALEMBIC_INI)


def _collect_ancestors(
    script_dir: ScriptDirectory, target_revision: str
) -> dict[str, pathlib.Path]:
    """
    Collect the set of all ancestor revision IDs up to and including target_revision
    via BFS, following down_revision links. Handles both linear chains and DAG
    branch merge points (where down_revision is a tuple).
    """
    try:
        root = script_dir.get_revision(target_revision)
    except Exception:
        root = None
    if root is None:
        raise ValueError(
            f"Revision {target_revision!r} not found in migration scripts."
        )

    ancestors: dict[str, pathlib.Path] = {}
    to_visit = [target_revision]

    while to_visit:
        revision_id = to_visit.pop()
        if revision_id in ancestors:
            continue

        revision = script_dir.get_revision(revision_id)
        ancestors[revision_id] = pathlib.Path(revision.path)

        if revision.down_revision is None:
            pass
        elif isinstance(revision.down_revision, str):
            to_visit.append(revision.down_revision)
        else:
            to_visit.extend(revision.down_revision)

    return ancestors


def _normalize_column_type(
    col_type: sqlalchemy.sql.type_api.TypeEngine,
) -> sqlalchemy.sql.type_api.TypeEngine:
    """Convert a reflected MySQL column type to an appropriate type.

    Rules:
    - TINYINT(1)        → Boolean  (MySQL stores BOOLEAN as TINYINT(1))
    - DATETIME/TIMESTAMP → mysql.DATETIME(fsp=N), preserving fsp so that env.py's
                           compare_type matches framework.db.sqldb.sql_types.DateTime;
                           TIMESTAMP is normalised to DATETIME for consistency with
                           what DateTime.load_dialect_impl creates on MySQL.
    - MEDIUMBLOB        → kept as-is; Blob() resolves to MEDIUMBLOB on MySQL
    - Everything else   → as_generic() to strip unnecessary dialect options
    """
    from sqlalchemy.dialects import mysql as mysql_types

    if (
        isinstance(col_type, mysql_types.TINYINT)
        and getattr(col_type, "display_width", None) == 1
    ):
        return sqlalchemy.Boolean()

    if isinstance(col_type, (mysql_types.DATETIME, mysql_types.TIMESTAMP)):
        fsp = getattr(col_type, "fsp", None)
        return mysql_types.DATETIME(fsp=fsp)

    if isinstance(col_type, mysql_types.MEDIUMBLOB):
        return col_type

    return col_type.as_generic() if hasattr(col_type, "as_generic") else col_type


def _reflect_schema(engine: sqlalchemy.engine.Engine) -> sqlalchemy.MetaData:
    """Reflect the full schema from the DB. Call this BEFORE dropping tables."""
    meta = sqlalchemy.MetaData()

    with engine.connect() as conn:
        meta.reflect(conn)

    # Remove alembic_version — alembic manages this table internally
    if "alembic_version" in meta.tables:
        meta.remove(meta.tables["alembic_version"])

    # Normalize column types and strip dialect-specific table options
    for table in meta.tables.values():
        for col in table.columns:
            col.type = _normalize_column_type(col.type)
            if isinstance(col.type, sqlalchemy.Boolean):
                col.autoincrement = "auto"
        for dialect_options in table.dialect_options.values():
            dialect_options._non_defaults.clear()

    return meta


def _drop_all_tables(engine: sqlalchemy.engine.Engine) -> None:
    """Drop all tables from the DB, disabling FK checks to avoid ordering issues."""
    insp = sqlalchemy.inspect(engine)
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("SET FOREIGN_KEY_CHECKS = 0"))
        for tname in insp.get_table_names():
            conn.execute(sqlalchemy.text(f"DROP TABLE `{tname}`"))
        conn.execute(sqlalchemy.text("SET FOREIGN_KEY_CHECKS = 1"))


def _generate_migration_code(
    engine: sqlalchemy.engine.Engine,
    reflected: sqlalchemy.MetaData,
) -> tuple[str, str, str]:
    """
    Compare an empty DB against the reflected schema to produce CREATE TABLE ops.

    Returns (upgrade_code, downgrade_code, imports) where:
    - upgrade_code: function body for upgrade() — CREATE TABLE ops
    - downgrade_code: function body for downgrade() — DROP TABLE ops
    - imports: module-level import lines beyond the standard alembic/sa imports
    """
    with engine.connect() as conn:
        mc = MigrationContext.configure(
            conn,
            opts={
                # compare_type suppresses spurious ALTER diffs for incremental migrations.
                # Here we generate pure CREATE ops from an empty DB, so it's not needed.
                "compare_type": None,
                "render_as_batch": False,
            },
        )
        migration = produce_migrations(mc, reflected)

    assert migration.upgrade_ops is not None
    assert migration.downgrade_ops is not None

    upgrade_code = render_python_code(migration.upgrade_ops, migration_context=mc)
    # The autogenerated migration for some reason will want to drop foreign
    # key indexes which will cause a crash, since the indexes will get deleted
    # by when table gets deleted anyway and we will delete every table we just
    # filter these operations out.
    downgrade_code = render_python_code(
        DowngradeOps(list(_skip_drop_index(migration.downgrade_ops.ops))),
        migration_context=mc,
    )

    # MySQL dialect types kept as-is (e.g. mysql.DATETIME, mysql.MEDIUMBLOB) are
    # rendered into upgrade_code but alembic does not auto-populate migration.imports
    # for them — add the import explicitly.
    migration.imports.add("from sqlalchemy.dialects import mysql")
    imports = "\n".join(sorted(migration.imports))
    return upgrade_code, downgrade_code, imports


def _skip_drop_index(ops: Iterable[MigrateOperation]) -> Iterator[MigrateOperation]:
    for op in ops:
        if isinstance(op, DropIndexOp):
            pass
        elif isinstance(op, ModifyTableOps):
            yield ModifyTableOps(
                op.table_name, list(_skip_drop_index(op.ops)), schema=op.schema
            )
        else:
            yield op


def main(target_revision: str, message: str) -> None:
    print(f"Squashing migrations up to and including: {target_revision}")

    # Step 1: Load config
    cfg = _load_config()

    # Step 2: Collect all ancestors via BFS; fail fast before touching the DB
    script_dir = ScriptDirectory.from_config(cfg)
    migration_files = _collect_ancestors(script_dir, target_revision)
    print(f"Found {len(migration_files)} migrations to squash")

    if len(migration_files) == 1:
        print(
            f"Target revision {target_revision!r} is already the root with no prior "
            "migrations. Nothing to squash.",
            file=sys.stderr,
        )
        sys.exit(0)

    # Step 3: Upgrade DB to target revision
    print(f"Running alembic upgrade to {target_revision}...")
    command.upgrade(cfg, target_revision)
    print("Upgrade complete.")

    # Step 4: Reflect schema BEFORE dropping
    dsn = cfg.get_main_option("sqlalchemy.url")
    engine = sqlalchemy.create_engine(dsn, poolclass=sqlalchemy.pool.NullPool)
    print("Reflecting schema from database...")
    reflected = _reflect_schema(engine)
    print(f"Reflected {len(reflected.tables)} tables: {sorted(reflected.tables)}")

    # Step 5: Drop all tables
    print("Dropping all tables...")
    _drop_all_tables(engine)
    print("Tables dropped.")

    # Step 6: Generate upgrade and downgrade code
    print("Generating migration code...")
    upgrade_code, downgrade_code, imports = _generate_migration_code(engine, reflected)
    engine.dispose()

    # Step 7: Delete old migration files
    print(f"Deleting {len(migration_files)} old migration files...")
    for path in migration_files.values():
        path.unlink()
        print(f"  Deleted: {path.name}")

    # Step 8: Write squashed migration using the existing Mako template.
    # Re-instantiate ScriptDirectory after deletion so it sees the updated state.
    script_dir2 = ScriptDirectory.from_config(cfg)
    script_dir2.generate_revision(
        target_revision,
        message,
        head=(),  # down_revision = None (new root)
        splice=True,  # allow creating a revision not descending from current head
        upgrades=upgrade_code,
        downgrades=downgrade_code,
        imports=imports,
    )
    squashed_path = pathlib.Path(script_dir2.get_revision(target_revision).path)

    print()
    print("=" * 60)
    print("Squash complete.")
    print(f"New file: {squashed_path}")
    print()
    print("Next steps:")
    print("  1. Review the generated migration file for correctness.")
    print(
        "  2. Verify that any migrations after this revision still have "
        f"     down_revision = {target_revision!r} (they should be unchanged)."
    )
    print("  3. Test: apply to a fresh DB with `alembic upgrade head`.")
    print()
    print("WARNING: DBs at a revision that was squashed away will fail to")
    print("  upgrade. Squashing is safe only when all deployed instances are")
    print(f"  at {target_revision!r} or a later revision.")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <target_revision> <message>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
