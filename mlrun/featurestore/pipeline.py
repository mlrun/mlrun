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
    WriteToTSDB,
)

from .model import TargetTypes
from .steps import ValidatorStep
from .targets import init_featureset_targets, add_target_states
from ..serving.server import MockContext
from ..serving.states import ServingFlowState


def init_graph(
    df,
    featureset,
    namespace,
    client=None,
    with_targets=False,
    return_df=True,
    verbose=True,
):
    context = MockContext()
    graph = featureset.spec.graph.copy()
    tables = {}
    targets = []

    if with_targets:
        init_featureset_targets(featureset, tables, True)
        targets = featureset.spec.targets
    add_target_states(graph, featureset, targets, to_df=return_df)

    def get_table(name):
        if name in tables:
            return tables[name]
        if name in ["", "."]:
            table = Table("", NoopDriver())
            tables[name] = table
            return table
        raise ValueError(f"table name={name} not set")

    def get_feature_set(uri):
        if uri in ["", "."]:
            return featureset
        if not client:
            raise ValueError("client must be set for remote features access")
        return client.get_feature_set(uri, use_cache=True)

    # enrich the context with classes and methods which will be used when
    # initializing classes or handling the event
    setattr(context, "get_feature_set", get_feature_set)
    setattr(context, "get_table", get_table)
    setattr(context, "current_function", "")
    setattr(context, "verbose", verbose)
    setattr(context, "root", graph)

    entity_columns = list(featureset.spec.entities.keys())
    if not entity_columns:
        raise ValueError("entity column(s) are not defined in feature set")
    key_column = entity_columns[0]
    source = DataframeSource(df, key_column, featureset.spec.timestamp_key)
    graph.set_flow_source(source)
    graph.init_object(context, namespace)
    return graph._controller


def steps_from_featureset(featureset, column_list, aliases, context):
    table = featureset.uri()
    entity_list = list(featureset.spec.entities.keys())
    key_column = entity_list[0]
    steps = []

    steps.append(
        QueryByKey(
            column_list, table, key=key_column, aliases=aliases, context=context,
        )
    )

    return steps
