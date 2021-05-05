from dateutil import parser

from mlrun.api.db.sqldb.models import Base, _table2cls
from mlrun.utils import get_in


def table2cls(name):
    return _table2cls.get(name)


def label_set(labels):
    if isinstance(labels, str):
        labels = labels.split(",")

    return set(labels or [])


def transform_label_list_to_dict(label_list):
    return {label.name: label.value for label in label_list}


def run_start_time(run):
    ts = get_in(run, "status.start_time", "")
    if not ts:
        return None
    return parser.parse(ts)


def run_labels(run) -> dict:
    return get_in(run, "metadata.labels", {})


def run_state(run):
    return get_in(run, "status.state", "")


def update_labels(obj, labels: dict):
    old = {label.name: label for label in obj.labels}
    obj.labels.clear()
    for name, value in labels.items():
        if name in old:
            old[name].value = value
            obj.labels.append(old[name])
        else:
            obj.labels.append(obj.Label(name=name, value=value, parent=obj.id))


def to_dict(obj):
    if isinstance(obj, Base):
        return {
            attr: to_dict(getattr(obj, attr)) for attr in dir(obj) if is_field(attr)
        }

    if isinstance(obj, (list, tuple)):
        cls = type(obj)
        return cls(to_dict(v) for v in obj)

    return obj


def is_field(name):
    if name[0] == "_":
        return False
    return name not in ("metadata", "Tag", "Label", "body")


def generate_query_predicate_for_name(column, query_string):
    if query_string.startswith("~"):
        return column.ilike(f"%{query_string[1:]}%")
    else:
        return column.__eq__(query_string)
