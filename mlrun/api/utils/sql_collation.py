from .mysql import MySQLUtil


class SQLCollationUtil(object):
    class Collations(object):

        # with sqlite we use the default collation
        sqlite = None
        mysql = "utf8_bin"

    @staticmethod
    def collation():
        mysql_dsn_data = MySQLUtil.get_mysql_dsn_data()
        if mysql_dsn_data:
            return SQLCollationUtil.Collations.mysql

        return SQLCollationUtil.Collations.sqlite
