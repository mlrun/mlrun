from storey import DataframeSource, FieldAggregator, AggregateByKey, Cache, V3ioDriver, NoopDriver, ReduceToDataFrame, \
    build_flow, Map, Persist
from storey.dtypes import SlidingWindows, FixedWindows

from .model import TargetTypes
from mlrun.serving.states import ServingTaskState


def _get_target_path(client, kind, featureset):
    name = featureset.metadata.name
    version = featureset.metadata.tag
    project = featureset.metadata.project or client.project or "default"
    if version:
        name = f"{name}-{version}"
    return f"{client.data_prefix}/{project}/{kind}/{name}"


def create_ingest_pipeline(client, featureset, source, targets=None,
                           namespace=[], return_df=True):

    targets = targets or []
    if TargetTypes.nosql in targets:
        target_path = _get_target_path(client, TargetTypes.nosql, featureset)
        cache = Cache(target_path, V3ioDriver())
    else:
        cache = Cache('', NoopDriver())

    key_column = featureset.spec.entities[0].name
    steps = [source]

    for state in featureset.spec._flow.values():
        steps.append(state_to_flow_object(state, context=None, namespace=namespace))

    aggregations = featureset.spec.aggregations.values()
    aggregation_objects = []
    for aggregate in aggregations:
        if aggregate.period:
            windows = SlidingWindows(aggregate.windows, aggregate.period)
        else:
            windows = FixedWindows(aggregate.windows)

        aggregator = FieldAggregator(aggregate.name,
                                     aggregate.column,
                                     aggregate.operations,
                                     windows)
        aggregation_objects.append(aggregator)

    if aggregation_objects:
        steps.append(AggregateByKey(aggregation_objects, cache))

    target_states = []
    if TargetTypes.nosql in targets:
        target_states.append(Persist(cache))
    if return_df:
        target_states.append(
            ReduceToDataFrame(index=key_column,
                              insert_key_column_as=key_column,
                              insert_time_column_as=featureset.spec.timestamp_key))

    if len(target_states) == 0:
        raise ValueError('must have at least one target or output df')
    if len(target_states) == 1:
        target_states = target_states[0]
    steps.append(target_states)
    return build_flow(steps)


def state_to_flow_object(state: ServingTaskState, context=None, namespace=[]):
    if not state.object:
        state.skip_context = True
        state.init_object(context, namespace)

    return state.object


