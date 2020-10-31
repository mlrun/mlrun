from storey import (
    JoinWithV3IOTable,
    V3ioDriver,
    QueryAggregationByKey,
    Cache,
    Map,
    Source,
    FieldAggregator,
    build_flow,
    Reduce,
)
from storey.dtypes import SlidingWindows
from .featureset import FeatureSet
from .model import TargetTypes


def steps_from_featureset(featureset: FeatureSet):
    def join_event(event, data):
        event.update(data)
        return event

    target = featureset.status.targets[TargetTypes.nosql]
    cache = Cache(target.path, V3ioDriver())
    column_list = list(featureset.spec.features.keys())
    entity_list = list(featureset.spec.entities.keys())
    key_column = entity_list[0]
    steps = []

    aggregations = featureset.spec.aggregations
    if aggregations:
        aggregation_objects = []
        for name, aggregate in aggregations.items():
            column_list = [col for col in column_list if not col.startswith(name + "_")]
            aggregator = FieldAggregator(
                name,
                aggregate.column,
                aggregate.operations,
                SlidingWindows(aggregate.windows, aggregate.period),
            )
            aggregation_objects.append(aggregator)

        steps.append(QueryAggregationByKey(aggregation_objects, cache, key_column))

    column_list.append("__name")
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


class QueryInput:
    def __init__(self, kv_table, columns, key_col, aliases):
        self.kv_table = kv_table
        self.columns = columns
        self.key_column = key_col
        self.aliases = aliases

    def get_steps(self):
        """
        Returns a list of storey steps represented by this object.
        First step would be either a `JoinWithV3IOTable` or `QueryAggregationByKey`.
        An optional second step would be a `Map` step for the aliases
        :return: list of steps
        """

        def join_event(event, data):
            event.update(data)
            return event

        steps = []
        if isinstance(self.columns[0], str):
            steps.append(
                JoinWithV3IOTable(
                    V3ioDriver(),
                    lambda x: x[self.key_column],
                    join_event,
                    self.kv_table,
                )
            )
        else:
            cache = Cache(self.kv_table, V3ioDriver())
            steps.append(QueryAggregationByKey(self.columns, cache, self.key_column))
        if self.aliases:

            def alias(event):
                for key, value in self.aliases.items():
                    event[value] = event.pop(key)
                return event

            steps.append(Map(alias))
        return steps


def build_query(queryInputs):
    steps = [Source()]
    for queries in queryInputs:
        steps.extend(queries.get_steps())
    steps.append(Reduce([], lambda acc, x: append_return(acc, x)))
    return build_flow(steps)


def append_return(lst, x):
    lst.append(x)
    return lst


def test_build_query():
    # aggr_test/ajCRMtLOms
    params = [
        QueryInput(
            "bigdata/aggr_test/ajCRMtLOms",
            [
                FieldAggregator(
                    "number_of_stuff",
                    "col1",
                    ["sum", "count"],
                    SlidingWindows(["1h", "24h"], "10m"),
                )
            ],
            "name",
            None,
        ),
        QueryInput(
            "bigdata/users_metadata", ["city", "age"], "name", {"city": "hometown"}
        ),
    ]
    controller = build_query(params).run()
    controller.emit({"name": "tal"})
    controller.terminate()
    res = controller.await_termination()
    print(res)
