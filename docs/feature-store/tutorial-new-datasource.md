# Develop new data source

Tutorial, how you can develop relation to the new data source. You can see full description
for these cases:
 - New off-line Source (such as addition new sources NoSQL, NewSQL, etc.)
 - New on-line Source (such as addition new streaming sources RabbitMQ, KQL, etc.)
 - New off-line Target
 - New on-line Target 

## New off-line Source
You have to follow next steps for develop new Source:

### Create new class derived from `BaseSourceDriver` 
1. Choose supported engines e.g. `storey`, `spark` or `pandas`, see the setting of variables
    support_storey = True
    support_spark = True
2. Implement method `to_step` (description see ...)
3. Implement method `to_dataframe` (description see ...)
4. Implement method `get_spark_options` in case of spark engine support
5. Implement method `to_spark_df` in case of spark engine support
6. Implement method `is_iterator`
7. ...

## New on-line Source
You have to follow next steps for develop new Source:

### Create new class derived from `OnlineSource`
1. ss
2. ff
3. 


## New on/off-line Target
You have to follow next steps for develop new Target:
1. aa
2. bb


NOTE: Typicall use case, write data to the Target (which was asociated with feature set). 


