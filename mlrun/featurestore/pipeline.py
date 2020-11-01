from storey import (
    FieldAggregator,
    AggregateByKey,
    Cache,
    V3ioDriver,
    NoopDriver,
    ReduceToDataFrame,
    build_flow,
    Persist,
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
        target_path = client._get_target_path(TargetTypes.parquet, featureset)
        target_path = upload_file(client, df, target_path, featureset)
        target = DataTarget(TargetTypes.parquet, target_path)
        featureset.status.update_target(target)
    client.save_object(featureset)
    return df


def process_to_df(df, featureset, entity_column, namespace):
    source = DataframeSource(df, entity_column, featureset.spec.timestamp_key)
    flow = create_ingest_pipeline(None, featureset, source, namespace=namespace)
    controller = flow.run()
    return controller.await_termination()


def enrich(event, state):
    state["xx"] = event["xx"]
    event["yy"] = 7
    state["yy"] = 7
    return event, state


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
        cache = Cache(target_path, V3ioDriver())
        target = DataTarget(TargetTypes.nosql, target_path)
        featureset.status.update_target(target)
    else:
        cache = Cache("", NoopDriver())

    key_column = featureset.spec.entities[0].name
    aggregations = featureset.spec.aggregations
    steps = [source]

    for state in featureset.spec.flow.states.values():
        steps.append(state_to_flow_object(state, context=None, namespace=namespace))

    if TargetTypes.nosql in targets:
        column_list = _clear_aggregators(aggregations, featureset.spec.features.keys())
        updater = UpdateState(column_list)
        steps.append(MapWithState(cache, updater.do, group_by_key=True))

    aggregation_objects = []
    for aggregate in aggregations.values():
        aggregation_objects.append(state_to_field_aggregator(aggregate))

    if aggregation_objects:
        steps.append(AggregateByKey(aggregation_objects, cache))

    target_states = []
    if TargetTypes.nosql in targets:
        target_states.append(Persist(cache))
    # if TargetTypes.parquet in targets:
    #     target_path = client._get_target_path(TargetTypes.parquet, featureset)
    #     target_states.append(WriteToParquet(target_path + '.parquet', partition_cols=[key_column]))
    #     target = DataTarget(TargetTypes.parquet, target_path)
    #     featureset.status.update_target(target)
    #
    if return_df:
        target_states.append(
            ReduceToDataFrame(
                index=key_column,
                insert_key_column_as=key_column,
                insert_time_column_as=featureset.spec.timestamp_key,
            )
        )

    if len(target_states) == 0:
        raise ValueError("must have at least one target or output df")
    if len(target_states) == 1:
        target_states = target_states[0]
    steps.append(target_states)
    return build_flow(steps)


def state_to_flow_object(state: ServingTaskState, context=None, namespace=[]):
    state._object = None
    if not state.object:
        # state.skip_context = True
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


def _clear_aggregators(aggregations, column_list):
    for name in aggregations.keys():
        column_list = [col for col in column_list if not col.startswith(name + "_")]
    return column_list


def steps_from_featureset(featureset, column_list):
    def join_event(event, data):
        event.update(data)
        return event

    target = featureset.status.targets[TargetTypes.nosql]
    cache = Cache(target.path, V3ioDriver())
    entity_list = list(featureset.spec.entities.keys())
    key_column = entity_list[0]
    steps = []

    aggregations = featureset.spec.aggregations
    if aggregations:
        aggregation_objects = []
        for aggregate in aggregations.values():
            aggregation_objects.append(state_to_field_aggregator(aggregate))
        steps.append(QueryAggregationByKey(aggregation_objects, cache, key_column))
        column_list = _clear_aggregators(aggregations, column_list)

    steps.append(
        JoinWithV3IOTable(
            V3ioDriver(),
            lambda x: x[key_column],
            join_event,
            target.path,
            attributes=column_list,
        )
    )
    return steps
