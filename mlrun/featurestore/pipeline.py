from storey import (
    Table,
    Driver,
    build_flow,
    DataframeSource,
    QueryByKey,
    Complete,
    Source,
)

from .targets import init_featureset_targets, add_target_states, get_online_target
from ..data_resources import ResourceCache
from ..serving.server import GraphContext, create_graph_server


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


def new_graph_context(tables, client=None, default_featureset=None):
    def get_table(name):
        if name in tables:
            return tables[name]
        if name in ["", "."]:
            table = Table("", Driver())
            tables[name] = table
            return table
        raise ValueError(f"table name={name} not set")

    def get_feature_set(uri):
        if uri in ["", "."]:
            return default_featureset
        if not client:
            raise ValueError("client must be set for remote features access")
        return client.get_feature_set(uri, use_cache=True)

    # enrich the context with classes and methods which will be used when
    # initializing classes or handling the event
    context = GraphContext()
    setattr(context, "get_feature_set", get_feature_set)
    setattr(context, "get_table", get_table)
    setattr(context, "current_function", "")
    return context


def print_event(event):
    print("EVENT:", str(event.key))
    print(str(event.body))
    return event


def init_feature_vector_graph(client, feature_set_fields, feature_set_objects):
    tables = {}
    context = new_graph_context(tables, client)
    steps = [Source()]
    for name, columns in feature_set_fields.items():
        fs = feature_set_objects[name]
        target, driver = get_online_target(fs)

        tables[fs.uri()] = driver.get_table_object()
        column_names = [name for name, alias in columns]
        aliases = {name: alias for name, alias in columns if alias}
        steps.extend(steps_from_featureset(fs, column_names, aliases, context))
    # steps.append(Map(print_event, full_event=True))
    steps.append(Complete())
    return build_flow(steps)


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
