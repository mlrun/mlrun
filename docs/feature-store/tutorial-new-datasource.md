# Develop new data source

Tutorial, how you can develop relation to the new data source. You can see full description
for these cases:
 
**Get data**
 1. New off-line Source (such as addition new sources NoSQL, NewSQL, etc.)
 2. New on-line Source (such as addition new streaming sources RabbitMQ, KQL, etc.)

**Write data**
 3. New off-line Target
 4. New on-line Target 

**Query data** via FeatureVector
 5. Support for query data ???

## 1. New off-line Source
You have to follow next steps for develop new Source:

### Create new class derived from `BaseSourceDriver` 
1. Choose supported engines e.g. `storey`, `spark` or `pandas`, see the setting of variables
    
    `support_storey = True`

    `support_spark = True`

2. Define source kind e.g. `xyz`

    `kind = "xyz"`
2. Implement method `to_step` (description of method/params see ...)
3. Implement method `to_dataframe` (description of method/params see ...)
4. Implement method `get_spark_options` in case of spark engine support (description of method/params see ...)
5. Implement method `to_spark_df` in case of spark engine support (description of method/params see ...)
6. Implement method `is_iterator` in case of chunk/bulk approach (description of method/params see ...)
7. Map of sources, add new item to the variable `mlrun.datastore.source.source_kind_to_driver`
8. ...

NOTE: Class `BaseSourceDriver` is derived from `DataSource`


## 2. New on-line Source
You have to follow next steps for develop new Source:

### Create new class derived from `OnlineSource`
1. Define source kind e.g. `xyz`

    `kind = "xyz"`
2. Implement method `add_nuclio_trigger` (description of method/params see ...)
3. Implement method `to_dataframe` (description of method/params see ...)
4. Implement method `to_spark_df` (description of method/params see ...)
5. Map of sources, add new item to the variable `mlrun.datastore.source.source_kind_to_driver`
6. ...

NOTE: Class `OnlineSource` is derived from `BaseSourceDriver`

## 3. New off-line Target
You have to follow next steps for develop new Target:

### Create new class derived from `BaseStoreTarget`
1. Choose supported for the setting

    `is_table = True`

    `is_offline = True`

    `support_spark = True`

    `support_storey = True`

    `support_dask = True`

    `support_append = True`
2. Define source kind e.g. `xyz`

    `kind = "xyz"`
3. Implement method `add_writer_step` (description of method/params see ...)
4. Implement method `as_df` (description of method/params see ...)
5. Implement method `is_single_file` (description of method/params see ...)
6. Implement method `get_spark_options` in case of spark engine support (description of method/params see ...)
7. Implement method `prepare_spark_df` in case of spark engine support(description of method/params see ...)
8. Map of target, add new item to the variable `mlrun.datastore.target.kind_to_driver`
9. ...

NOTE: Class `BaseStoreTarget` is derived from `DataTargetBase`

## 4. New off-line Target

TBD.

## 5. Support for query data ???