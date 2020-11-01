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
        None, featureset, source, namespace=namespace  # , return_df=return_df
    )
    controller = flow.run()
    df = controller.await_termination()
    if TargetTypes.parquet in targets:
        target_path = client._get_target_path(TargetTypes.parquet, featureset)
        target_path = upload_file(client, df, target_path, featureset, format="csv")
        print("path:", target_path)
        target = DataTarget(TargetTypes.parquet, target_path)
        featureset.status.update_target(target)
    client.save_object(featureset)
    return df


def process_to_df(df, featureset, entity_column, namespace):
    source = DataframeSource(df, entity_column, featureset.spec.timestamp_key)
    flow = create_ingest_pipeline(None, featureset, source, namespace=namespace)
    controller = flow.run()
    return controller.await_termination()


def create_ingest_pipeline(
    client, featureset, source, targets=None, namespace=[], return_df=True
):

    targets = targets or []
    if TargetTypes.nosql in targets:
        target_path = client._get_target_path(TargetTypes.nosql, featureset)
        cache = Cache(target_path, V3ioDriver())
        target = DataTarget(TargetTypes.nosql, target_path)
        featureset.status.update_target(target)
    else:
        cache = Cache("", NoopDriver())

    key_column = featureset.spec.entities[0].name
    steps = [source]

    for state in featureset.spec.flow.states.values():
        steps.append(state_to_flow_object(state, context=None, namespace=namespace))

    aggregations = featureset.spec.aggregations.values()
    aggregation_objects = []
    for aggregate in aggregations:
        if aggregate.period:
            windows = SlidingWindows(aggregate.windows, aggregate.period)
        else:
            windows = FixedWindows(aggregate.windows)

        aggregator = FieldAggregator(
            aggregate.name, aggregate.column, aggregate.operations, windows
        )
        aggregation_objects.append(aggregator)

    if aggregation_objects:
        steps.append(AggregateByKey(aggregation_objects, cache))

    target_states = []
    if TargetTypes.nosql in targets:
        target_states.append(Persist(cache))
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
    if not state.object:
        state.skip_context = True
        state.init_object(context, namespace)

    return state.object
