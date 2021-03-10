from v3io.dataplane.kv import Model


class MockKV(Model):
    def __init__(self):
        super().__init__(client=None)
        self.kv = {}

    def new_cursor(
        self,
        container,
        table_path,
        access_key=None,
        raise_for_status=None,
        attribute_names="*",
        filter_expression=None,
        marker=None,
        sharding_key=None,
        limit=None,
        segment=None,
        total_segments=None,
        sort_key_range_start=None,
        sort_key_range_end=None,
    ):
        raise NotImplementedError()

    def put(
        self,
        container,
        table_path,
        key,
        attributes,
        access_key=None,
        raise_for_status=None,
        transport_actions=None,
        condition=None,
    ):
        self.kv[key] = attributes

    def update(
        self,
        container,
        table_path,
        key,
        access_key=None,
        raise_for_status=None,
        transport_actions=None,
        attributes=None,
        expression=None,
        condition=None,
        update_mode=None,
        alternate_expression=None,
    ):
        if key not in self.kv:
            self.put(container, table_path, key, attributes)
        else:
            for k, v in self.kv[key].items():
                self.kv[key][k] = attributes.get(k, v)

    def get(
        self,
        container,
        table_path,
        key,
        access_key=None,
        raise_for_status=None,
        transport_actions=None,
        attribute_names="*",
    ):
        return self.kv[key]

    def scan(
        self,
        container,
        table_path,
        access_key=None,
        raise_for_status=None,
        transport_actions=None,
        attribute_names="*",
        filter_expression=None,
        marker=None,
        sharding_key=None,
        limit=None,
        segment=None,
        total_segments=None,
        sort_key_range_start=None,
        sort_key_range_end=None,
    ):
        raise NotImplementedError()

    def delete(
        self,
        container,
        table_path,
        key,
        access_key=None,
        raise_for_status=None,
        transport_actions=None,
    ):
        self.kv.pop(key)

    def create_schema(
        self,
        container,
        table_path,
        access_key=None,
        raise_for_status=None,
        transport_actions=None,
        key=None,
        fields=None,
    ):
        raise NotImplementedError()


class MockV3IOClient:
    def __init__(self):
        self.kv = MockKV()
