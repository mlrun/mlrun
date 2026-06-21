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

"""Squash alembic migrations into a single migration file.

Two modes are supported:

  root <target_revision> <message>
      Squash from root up to and including <target_revision> into a single new
      *root* migration (down_revision = None). The whole schema is reflected from
      an empty DB, so the result is pure CREATE TABLE ops.

  head <base_revision> <message>
      Squash every migration *after* <base_revision> up to the current head into
      a single delta migration whose down_revision = <base_revision>. The delta is
      produced by alembic autogenerate against the ORM models (migrations/env.py),
      which only matches the head when models == head schema — this is verified
      before squashing.

Must be run from server/py/services/api/ (where alembic.ini lives).
PYTHONPATH and MLRUN_HTTPDB__DSN must be set (done by squash_migrations_mysql.sh).
"""

import argparse
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


def _revision_exists(script_dir: ScriptDirectory, revision_id: str) -> bool:
    try:
        return script_dir.get_revision(revision_id) is not None
    except Exception:
        return False


def _collect_ancestors(
    script_dir: ScriptDirectory,
    target_revision: str,
    base_revision: str | None = None,
) -> dict[str, pathlib.Path]:
    """
    Collect the revision-id -> path map for the ancestors of target_revision via
    BFS, following down_revision links. Handles both linear chains and DAG branch
    merge points (where down_revision is a tuple).

    If base_revision is None, walk all the way to the root (inclusive). Otherwise
    base_revision acts as a wall: it is neither collected nor descended past, so
    only revisions strictly after base up to and including target are returned.
    In that case base must be an ancestor of target on *every* path — if the walk
    reaches a root migration without passing through base (e.g. base sits on only
    one side of a merge that target descends from), it is rejected.

    Raises if target_revision is not found, or if base_revision is given but is
    not found / not an ancestor of target_revision on every path.
    """
    if not _revision_exists(script_dir, target_revision):
        raise ValueError(
            f"Revision {target_revision!r} not found in migration scripts."
        )
    if base_revision is not None and not _revision_exists(script_dir, base_revision):
        raise ValueError(f"Base revision {base_revision!r} not found.")

    ancestors: dict[str, pathlib.Path] = {}
    to_visit = [target_revision]

    while to_visit:
        revision_id = to_visit.pop()
        if revision_id == base_revision:
            continue
        if revision_id in ancestors:
            continue

        revision = script_dir.get_revision(revision_id)
        ancestors[revision_id] = pathlib.Path(revision.path)

        # down_revision is None or () for a root, a str for a single parent, or a
        # tuple for a merge point — normalize to a tuple of parent revision ids.
        if revision.down_revision is None:
            down_revisions = ()
        elif isinstance(revision.down_revision, str):
            down_revisions = (revision.down_revision,)
        else:
            down_revisions = tuple(revision.down_revision)

        if not down_revisions:
            # A root migration. With a base set, every path must terminate at the
            # base wall; reaching a root means this path bypassed base.
            if base_revision is not None:
                raise ValueError(
                    f"Reached root migration {revision_id!r} without passing "
                    f"through base {base_revision!r}; base must be an ancestor of "
                    f"{target_revision!r} on every path."
                )
        else:
            to_visit.extend(down_revisions)

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


def squash_from_root(target_revision: str, message: str) -> None:
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


def _has_pending_changes(cfg: Config) -> bool:
    """Return True if autogenerate against the ORM models would produce any ops.

    Runs through migrations/env.py (so it uses the project's custom compare_type)
    and discards the directives without writing a file. The DB must already be at
    the script-directory head before calling.
    """
    captured: dict[str, bool] = {}

    def _capture(context, revision, directives):
        captured["empty"] = directives[0].upgrade_ops.is_empty()
        # Clear directives so alembic does not write a migration file.
        directives[:] = []

    command.revision(cfg, autogenerate=True, process_revision_directives=_capture)
    return not captured.get("empty", True)


def squash_to_head(base_revision: str, message: str) -> None:
    print(f"Squashing migrations after {base_revision!r} up to head")

    # Step 1: Load config and resolve the single head; fail fast before the DB.
    cfg = _load_config()
    script_dir = ScriptDirectory.from_config(cfg)

    heads = script_dir.get_heads()
    if len(heads) != 1:
        raise ValueError(
            f"Expected exactly one head, found {len(heads)}: {sorted(heads)}. "
            "Head squashing requires a single linear head."
        )
    old_head = heads[0]

    if base_revision == old_head:
        print(
            f"Base revision {base_revision!r} is already the head. Nothing to squash.",
            file=sys.stderr,
        )
        sys.exit(0)

    range_files = _collect_ancestors(script_dir, old_head, base_revision=base_revision)
    print(f"Found {len(range_files)} migrations to squash into one")

    # Step 2: Upgrade a fresh DB to head and confirm the models match it. The
    # squash derives its delta from the models, so if they differ from the head
    # schema the result would not reproduce head — abort rather than bake in drift.
    print(f"Upgrading to head {old_head} for drift check...")
    command.upgrade(cfg, old_head)
    if _has_pending_changes(cfg):
        print(
            "ERROR: ORM models differ from the head schema (autogenerate found "
            "pending changes). A head squash is derived from the models, so it "
            "would not reproduce the current head. Resolve the drift first.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("Models match head schema.")

    # Step 3: Rebuild the DB to the base revision (drop everything, upgrade to base).
    # env.py has now set sqlalchemy.url from mlconf, so this reads the real DSN.
    dsn = cfg.get_main_option("sqlalchemy.url")
    engine = sqlalchemy.create_engine(dsn, poolclass=sqlalchemy.pool.NullPool)
    print(f"Dropping all tables and rebuilding to base {base_revision}...")
    _drop_all_tables(engine)
    engine.dispose()
    command.upgrade(cfg, base_revision)
    print(f"DB is now at base {base_revision}.")

    # Step 4: Delete the squashed-away migration files so base becomes the head.
    print(f"Deleting {len(range_files)} old migration files...")
    for path in range_files.values():
        path.unlink()
        print(f"  Deleted: {path.name}")

    # Step 5: Autogenerate the squashed delta. With base as the sole head,
    # down_revision = base; reuse the old head revision id so the head id is stable.
    print("Generating squashed migration via autogenerate...")
    command.revision(cfg, message=message, autogenerate=True, rev_id=old_head)
    script_dir2 = ScriptDirectory.from_config(cfg)
    squashed_path = pathlib.Path(script_dir2.get_revision(old_head).path)

    print()
    print("=" * 60)
    print("Squash complete.")
    print(f"New file: {squashed_path}")
    print()
    print("Next steps:")
    print("  1. Review the generated migration file for correctness.")
    print(f"  2. Confirm it has down_revision = {base_revision!r} and is the head.")
    print("  3. Test: apply to a fresh DB with `alembic upgrade head` and verify")
    print("     the schema matches the pre-squash schema.")
    print()
    print("WARNING: the migrations between the base and the head are removed, so a")
    print("  DB at any of those squashed-away revisions will fail to upgrade.")
    print("  Squashing is safe only when every deployed instance is at")
    print(f"  {base_revision!r} or earlier (instances already at the head are")
    print("  unaffected, since its revision id is preserved).")
    print("=" * 60)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="mode", required=True)

    root_parser = subparsers.add_parser(
        "root",
        help="Squash from root up to and including a target revision into a new "
        "root migration.",
    )
    root_parser.add_argument("target_revision")
    root_parser.add_argument("message")
    root_parser.set_defaults(
        func=lambda args: squash_from_root(args.target_revision, args.message)
    )

    head_parser = subparsers.add_parser(
        "head",
        help="Squash all migrations after a base revision up to the current head "
        "into a single migration.",
    )
    head_parser.add_argument("base_revision")
    head_parser.add_argument("message")
    head_parser.set_defaults(
        func=lambda args: squash_to_head(args.base_revision, args.message)
    )

    return parser


if __name__ == "__main__":
    parsed_args = _build_parser().parse_args()
    parsed_args.func(parsed_args)
