from storey import DataframeSource

from mlrun.featurestore.targets import init_featureset_targets, add_target_states
from mlrun.datastore.data_resources import ResourceCache
from mlrun.serving.server import create_graph_server


def init_featureset_graph(
    df, featureset, namespace, with_targets=False, return_df=True,
):
    cache = ResourceCache()
    targets = []
    graph = featureset.spec.graph.copy()
    start_states, default_final_state, _ = graph.check_and_process_graph(
        allow_empty=True
    )

    # init targets (and table)
    if with_targets:
        table = init_featureset_targets(featureset)
        if table:
            cache.cache_table(featureset.uri(), table, True)
        targets = featureset.spec.targets

    cache.cache_resource(featureset.uri(), featureset, True)
    add_target_states(
        graph, featureset, targets, to_df=return_df, final_state=default_final_state
    )

    # init source
    entity_columns = list(featureset.spec.entities.keys())
    if not entity_columns:
        raise ValueError("entity column(s) are not defined in feature set")
    key_column = entity_columns[0]
    source = DataframeSource(df, key_column, featureset.spec.timestamp_key)
    graph.set_flow_source(source)

    server = create_graph_server(graph=graph, parameters={})
    server.init(None, namespace, cache)
    return graph._controller


def featureset_initializer(server):
    context = server.context
    cache = server.resource_cache
    featureset_uri = context.get_param("featureset")
    featureset = context.get_data_resource(featureset_uri)

    table = init_featureset_targets(featureset)
    if table:
        cache.cache_table(featureset.uri(), table, True)
    cache.cache_resource(featureset.uri(), featureset, True)

    targets = featureset.spec.targets
    add_target_states(server.graph, featureset, targets)

    # get source object from spec.source

    # set source
