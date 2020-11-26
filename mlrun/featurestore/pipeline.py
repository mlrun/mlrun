from storey import (
    AggregateByKey,
    Table,
    NoopDriver,
    ReduceToDataFrame,
    build_flow,
    WriteToTable,
    DataframeSource,
    WriteToParquet,
    QueryByKey,
)

from .model import TargetTypes


def ingest_from_df(
        context, featureset, df, targets=None, namespace=[], return_df=True
):
    entity_columns = list(featureset.spec.entities.keys())
    if not entity_columns:
        raise ValueError("entity column(s) are not defined in feature set")
    key_column = entity_columns[0]
    source = DataframeSource(df, key_column, featureset.spec.timestamp_key)

    return create_ingest_flow(context, featureset, source, targets, namespace=namespace, return_df=return_df).run()


def create_ingest_flow(
    context, featureset, source, targets=None, namespace=[], return_df=True
):

    key_column = featureset.spec.entities[0].name
    targets = targets or []
    table = featureset.uri() if TargetTypes.nosql in targets else Table("", NoopDriver())

    timestamp_key = featureset.spec.timestamp_key
    aggregations = featureset.spec.aggregations
    steps = [source]

    for state in featureset.spec.flow.states.values():
        state.clear_object()  # clear the state object due to storey limitations
        state.init_object(context, namespace)
        steps.append(state.object)

    column_list = _clear_aggregators(aggregations, featureset.spec.features.keys())
    aggregation_objects = [aggregate.to_dict() for aggregate in aggregations.values()]
    if aggregation_objects:
        steps.append(AggregateByKey(aggregation_objects, table, context=context))

    if TargetTypes.nosql in targets:
        steps.append([WriteToTable(table, columns=column_list, context=context)])

    if TargetTypes.parquet in targets:
        target_path = featureset.status.targets['parquet'].path
        column_list = list(featureset.spec.features.keys())
        if timestamp_key:
            column_list = [timestamp_key] + column_list
        steps.append([WriteToParquet(target_path, index_cols=key_column, columns=column_list)])

    if return_df:
        steps.append([
            ReduceToDataFrame(
                index=key_column,
                insert_key_column_as=key_column,
                insert_time_column_as=featureset.spec.timestamp_key,
            )]
        )

    return build_flow(steps)


def _clear_aggregators(aggregations, column_list):
    remove_list = []
    for name in aggregations.keys():
        for col in column_list:
            if col.startswith(name + "_"):
                remove_list.append(col)

    column_list = [col for col in column_list if col not in remove_list]
    return column_list


def steps_from_featureset(featureset, column_list, aliases, context):
    table = featureset.uri()
    entity_list = list(featureset.spec.entities.keys())
    key_column = entity_list[0]
    steps = []

    steps.append(
        QueryByKey(
            column_list,
            table,
            key=key_column,
            aliases=aliases,
            context=context,
        )
    )

    return steps
