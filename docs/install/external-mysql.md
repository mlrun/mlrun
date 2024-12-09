(external-mysql)=
# Configuring an external MySQL database

## Prerequisites
- MySQL Version: Ensure that you're using MySQL version 8.0. MLRun does not support MySQL 8.4 and higher.
- Database Creation: The MLRun database must exist in your MySQL instance. You can create it by running the following SQL command:
    ```
	CREATE DATABASE mlrun;
	```
- DSN Format: The MLRUN_HTTPDB__DSN (Data Source Name) should be formatted like:
   ```
   mysql+pymysql://<username>:<password>@<ip>:<port>/mlrun
   ```
 
## Configuration
1. Deploy your Iguazio system.
2. On each data node, navigate to the chart directory and unzip the MLRun chart:
    ```
	cd /home/iguazio/igz/platform/static_serve/helm/v3io-stable/
	tar xf mlrun-<latest-version>.tgz
	```
2. Update the configuration: Run the following command, replacing <REPLACE-DSN-HERE> with your actual DSN:
    ```
	MLRUN_HTTPDB__DSN="<REPLACE-DSN-HERE>" yq eval --inplace '(.api.extraEnvKeyValue.MLRUN_HTTPDB__DSN=strenv(MLRUN_HTTPDB__DSN)) | (.api.extraEnvKeyValue.MLRUN_HTTPDB__DB__BACKUP__MODE="disabled")' | (.api.extraEnvKeyValue.MLRUN_HTTPDB__DB__MYSQL__MODES="nil")' mlrun/values.yaml
	```

```{admonition} Notes
- To disable database backup by MLRun (MLRun will not attempt to perform database backups, meaning the responsobility for backups is outside of MLRun), run: `MLRUN_HTTPDB__DB__BACKUP__MODE="disabled"`
- SQL modes:
   - STRICT_TRANS_TABLES: Raises an error for invalid or out-of-range transaction values.
   - NO_ZERO_IN_DATE: Prevents dates with zeros in year, month, or day (e.g., '2023-00-01').
   - NO_ZERO_DATE: Disallows '0000-00-00' as a valid date.
   - ERROR_FOR_DIVISION_BY_ZERO: Returns an error for division by zero instead of NULL.
   - NO_ENGINE_SUBSTITUTION: Raises an error if a specified storage engine is unavailable.

   These modes enforce stricter data validation and error handling, ensuring better data integrity in MLRun operations.

- External database: When using an external MySQL database, SQL modes are disabled by setting `MLRUN_HTTPDB__DB__MYSQL__MODES="nil"`. You are responsible for configuring the SQL modes based on your needs.
```

2. (Optional) Backup the original file.
   If you need to retain a backup of the original chart before making changes, copy the chart file with:
   ```
   cp mlrun-<latest-version>.tgz mlrun-<latest-version>.tgz.bak
   ```
2. Repackage the chart:

   After the configuration is updated, you can repackage the chart using Helm:
   ```
   helm package --destination ./ ./mlrun
   ```
2. Restart MLRun in the UI services page, to apply the changes.
2. Verify the configuration:
   
   Ensure that the MLRun chief and workers have the following environment variables set to the new values:
   - MLRUN_HTTPDB__DSN: The correct Data Source Name for the external MySQL service.
   - MLRUN_HTTPDB__DB__BACKUP__MODE: Set to disabled.

## Azure database for MySQL configuration
If you're using Azure MySQL, pay particular attention to the default parameter settings that are configured for Azure's MySQL service, and modify tjem according to your needs.

- Increase `max_connections`:
   Azure MySQL has a default value for the `max_connections parameter`, which may not be sufficient for MLRun, especially in high-traffic scenarios.
   This default value can lead to a "Too many connections" error if the number of concurrent connections exceeds the limit. This is particularly common 
   in environments with heavy loads or multiple services interacting with the database.
   
   To avoid this issue and ensure the database can handle more connections, increase the `max_connections` parameter to a value that fits your expected workload.
   
- Disabling `ONLY_FULL_GROUP_BY` Mode:

   Azure MySQL comes with ONLY_FULL_GROUP_BY enabled by default. This SQL mode can cause issues for MLRun, as it might conflict with certain 
   queries used by MLRun that rely on non-standard SQL groupings.
   
    To prevent errors related to strict grouping behavior in queries and to ensure that MLRun works properly with your Azure MySQL instance, it's 
	recommended to disable the ONLY_FULL_GROUP_BY mode in your MySQL configuration. For detailed instructions, see the Microsoft documentation: 
   [How to disable ONLY_FULL_GROUP_BY mode - Microsoft Q&A](https://learn.microsoft.com/en-us/answers/questions/628390/how-to-disable-only-full-group-by-mode).

- Remove the Secure Transport Requirement:

   MLRun does not yet support connections to a secured (SSL-based) remote database directly. As a workaround, you can remove the secure 
   transport requirement for Azure MySQL. To do this, follow the steps in [Encrypted connectivity using TLS/SSL - Azure Database for MySQL - Flexible Server](https://learn.microsoft.com/en-us/azure/mysql/flexible-server/how-to-connect-tls-ssl#disable-ssl-enforcement-on-your-azure-database-for-mysql-flexible-server-instance), 
   which outlines how to configure encrypted connections using TLS/SSL.

   Removing the secure transport requirement allows MLRun to connect to the MySQL database without enforcing SSL encryption, 
   which is necessary until full support for SSL-based connections is implemented in MLRun.  

## AWS RDS Configuration

When configuring AWS RDS (Relational Database Service) for use with MLRun, you can choose from different types of RDS instances depending on 
your high availability and scalability requirements. MLRun supports the following AWS RDS configurations:
- Single DB Instance
- Multi-AZ DB Instance

```{admonition} Note
AWS Multi-AZ DB Cluster is not supported for MLRun at this time. <!-- ML-8165 -->
```

### Important parameter adjustments for AWS RDS
- Increase `innodb_write_io_threads`:
   By default, `innodb_write_io_threads` is set to 4 for many MySQL setups. Under heavy transactional loads, it is recommended to 
   increase this value to 8 or more. This enables MySQL to handle more I/O operations concurrently, reducing the likelihood of lock-wait timeouts.

- Adjust `innodb_read_io_threads`:
    If your workload involves high read-throughput, you might want to consider increasing the `innodb_read_io_threads` parameter to improve read I/O performance.

 