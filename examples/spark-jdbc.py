# Pyspark example called by mlrun_sparkk8s.ipynb
import os
# Iguazio env
V3IO_USER = os.getenv('V3IO_USERNAME')
V3IO_HOME = os.getenv('V3IO_HOME')
V3IO_HOME_URL = os.getenv('V3IO_HOME_URL')

os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages mysql:mysql-connector-java:5.1.39 pyspark-shell"

from pyspark.sql import SparkSession


spark = SparkSession.builder.    appName("Spark JDBC to Databases - ipynb").getOrCreate()



#Loading data from a JDBC source
dfMySQL = spark.read     .format("jdbc")     .option("url", "jdbc:mysql://mysql-rfam-public.ebi.ac.uk:4497/Rfam")     .option("dbtable", "Rfam.family")     .option("user", "rfamro")     .option("password", "")     .option("driver", "com.mysql.jdbc.Driver")     .load()


dfMySQL.write.format("io.iguaz.v3io.spark.sql.kv").mode("overwrite").option("key", "rfam_id").save("v3io://users/admin/frommysql")

