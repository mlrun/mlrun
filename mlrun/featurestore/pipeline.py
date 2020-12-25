from storey import (
    DataframeSource,
    Source,
)

from .targets import init_featureset_targets, add_target_states, get_online_target
from mlrun.datastore.data_resources import ResourceCache
from ..serving.server import create_graph_server


def init_featureset_graph(
    df, featureset, namespace, with_targets=False, return_df=True,
):
    cache = ResourceCache()
    targets = []
    graph = featureset.spec.graph.copy()

    # init targets (and table)
    if with_targets:
        table = init_featureset_targets(featureset)
        if table:
            cache.cache_table(featureset.uri(), table, True)
        targets = featureset.spec.targets

    cache.cache_resource(featureset.uri(), featureset, True)
    add_target_states(graph, featureset, targets, to_df=return_df)

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


def _build_feature_vector_graph(
    vector, feature_set_fields, feature_set_objects,
):
    graph = vector.spec.graph.copy()
    start_at = graph.start_at
    next = graph

    for name, columns in feature_set_fields.items():
        featureset = feature_set_objects[name]
        column_names = [name for name, alias in columns]
        aliases = {name: alias for name, alias in columns if alias}

        entity_list = list(featureset.spec.entities.keys())
        key_column = entity_list[0]
        next = next.to(
            "storey.QueryByKey",
            f"query-{name}",
            features=column_names,
            table=featureset.uri(),
            key=key_column,
            aliases=aliases,
        )
    if start_at:
        next.set_next(start_at)
    last_state = graph.find_last_state()
    if not last_state:
        raise ValueError("the graph doesnt have an explicit final step to respond on")
    graph[last_state].respond()

    return graph


def init_feature_vector_graph(vector):
    feature_set_objects, feature_set_fields = vector.parse_features()
    graph = _build_feature_vector_graph(vector, feature_set_fields, feature_set_objects)
    graph.set_flow_source(Source())
    server = create_graph_server(graph=graph, parameters={})

    cache = ResourceCache()
    for featureset in feature_set_objects.values():
        target, driver = get_online_target(featureset)
        cache.cache_table(featureset.uri(), driver.get_table_object())
    server.init(None, None, cache)
    return graph._controller
