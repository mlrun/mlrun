from storey import (
    FieldAggregator,
    AggregateByKey,
    Table,
    V3ioDriver,
    NoopDriver,
    ReduceToDataFrame,
    build_flow,
    WriteToTable,
    DataframeSource,
    WriteToParquet,
    MapWithState,
    QueryAggregationByKey,
    JoinWithV3IOTable,
)
from storey.dtypes import SlidingWindows, FixedWindows

from .ingest import upload_file
from .model import TargetTypes, DataTarget
from mlrun.serving.states import ServingTaskState


def run_ingestion_pipeline(
    client, featureset, source, targets=None, namespace=[], return_df=True
):
    if not targets:
        raise ValueError("ingestion target(s) were not specified")

    if isinstance(source, str):
        # if source is a path/url convert to DataFrame
        source = client.get_data_stores().object(url=source).as_df()

    entity_columns = list(featureset.spec.entities.keys())
    if not entity_columns:
        raise ValueError("entity column(s) are not defined in feature set")
    source = DataframeSource(source, entity_columns[0], featureset.spec.timestamp_key)

    flow = create_ingest_pipeline(
        client,
        featureset,
        source,
        targets,
        namespace=namespace,  # , return_df=return_df
    )
    controller = flow.run()
    df = controller.await_termination()
    if TargetTypes.parquet in targets:
        target_path = client._get_target_path(TargetTypes.parquet, featureset, '.parquet')
        target_path = upload_file(client, df, target_path, featureset)
        target = DataTarget(TargetTypes.parquet, target_path)
        featureset.status.update_target(target)
    return df


def process_to_df(df, featureset, entity_column, namespace):
    source = DataframeSource(df, entity_column, featureset.spec.timestamp_key)
    flow = create_ingest_pipeline(None, featureset, source, namespace=namespace)
    controller = flow.run()
    return controller.await_termination()


class UpdateState:
    def __init__(self, fields):
        self.fields = fields

    def do(self, event, state):
        for field in self.fields:
            value = event.get(field, None)
            if value is not None:
                state[field] = value
        return event, state


def create_ingest_pipeline(
    client, featureset, source, targets=None, namespace=[], return_df=True
):

    targets = targets or []
    if TargetTypes.nosql in targets:
        target_path = client._get_target_path(TargetTypes.nosql, featureset, "/")
        table = Table(target_path, V3ioDriver())
        target = DataTarget(TargetTypes.nosql, target_path)
        featureset.status.update_target(target)
    else:
        table = Table("", NoopDriver())

    key_column = featureset.spec.entities[0].name
    aggregations = featureset.spec.aggregations
    steps = [source]

    for state in featureset.spec.flow.states.values():
        steps.append(state_to_flow_object(state, context=None, namespace=namespace))

    column_list, _ = _clear_aggregators(aggregations, featureset.spec.features.keys())
    if TargetTypes.nosql in targets:
        updater = UpdateState(column_list)
        steps.append(MapWithState(table, updater.do, group_by_key=True))

    aggregation_objects = []
    for aggregate in aggregations.values():
        aggregation_objects.append(state_to_field_aggregator(aggregate))

    if aggregation_objects:
        steps.append(AggregateByKey(aggregation_objects, table))

    if TargetTypes.nosql in targets:
        steps.append([WriteToTable(table)])
    # if TargetTypes.parquet in targets:
    #     print('KEY:', key_column)
    #     target_path = client._get_target_path(TargetTypes.parquet, featureset, '.parquet')
    #     steps.append([WriteToParquet(target_path, metadata_columns={key_column: 'key'})])
    #     target = DataTarget(TargetTypes.parquet, target_path)
    #     featureset.status.update_target(target)

    if return_df:
        steps.append([
            ReduceToDataFrame(
                index=key_column,
                insert_key_column_as=key_column,
                insert_time_column_as=featureset.spec.timestamp_key,
            )]
        )

    return build_flow(steps)


def state_to_flow_object(state: ServingTaskState, context=None, namespace=[]):
    state.clear_object()  # clear the state object due to storey limitations
    if not state.object:
        state.init_object(context, namespace)

    return state.object


def state_to_field_aggregator(aggregate):
    if aggregate.period:
        windows = SlidingWindows(aggregate.windows, aggregate.period)
    else:
        windows = FixedWindows(aggregate.windows)

    return FieldAggregator(
        aggregate.name, aggregate.column, aggregate.operations, windows
    )


def _clear_aggregators2(aggregations, column_list):
    for name in aggregations.keys():
        column_list = [col for col in column_list if not col.startswith(name + "_")]
    return column_list


def _clear_aggregators(aggregations, column_list):
    aggregates_list = []
    for name, aggregate in aggregations.items():
        remove_list = []
        operations = []
        windows = []
        for col in column_list:
            if col.startswith(name + "_"):
                split = col[len(name + "_"):].split('_')
                if len(split) != 2:
                    raise ValueError(f'illegal aggregation column {col}, '
                                     'must be in the form {name}_{op}_{window}')
                if split[0] not in aggregate.operations:
                    raise ValueError(f'aggregate operation {split[0]} doesnt exist')
                if split[0] not in operations:
                    operations.append(split[0])

                if split[1] not in aggregate.windows:
                    raise ValueError(f'aggregate window {split[1]} doesnt exist')
                if split[1] not in windows:
                    windows.append(split[1])
                remove_list.append(col)
        if operations:
            aggregate = aggregate.copy()
            aggregate.operations = operations
            aggregate.windows = windows
            aggregates_list.append(aggregate)

        column_list = [col for col in column_list if col not in remove_list]
    return column_list, aggregates_list


def steps_from_featureset(featureset, column_list, aliases):
    target = featureset.status.targets[TargetTypes.nosql]
    table = Table(target.path, V3ioDriver())
    entity_list = list(featureset.spec.entities.keys())
    key_column = entity_list[0]
    steps = []

    aggregations = featureset.spec.aggregations
    if aggregations:
        aggregation_objects = []
        column_list, aggregates_list = _clear_aggregators(aggregations, column_list)
        for aggregate in aggregates_list:
            aggregation_objects.append(state_to_field_aggregator(aggregate))

        steps.append(
            QueryAggregationByKey(
                aggregation_objects,
                table,
                key=key_column,
                enrich_with=column_list,
                aliases=aliases,
            )
        )

    return steps
