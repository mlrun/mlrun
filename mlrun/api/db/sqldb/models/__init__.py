from mlrun.api.utils.db.mysql import MySQLUtil

# fmt: off
mysql_dsn_data = MySQLUtil.get_mysql_dsn_data()
if mysql_dsn_data:
    from .models_mysql import *  # noqa

    # importing private variables as well
    from .models_mysql import _classes, _labeled, _table2cls, _tagged  # noqa # isort:skip
else:
    from .models_sqlite import *  # noqa

    # importing private variables as well
    from .models_sqlite import _classes, _labeled, _table2cls, _tagged  # noqa # isort:skip
# fmt: on
