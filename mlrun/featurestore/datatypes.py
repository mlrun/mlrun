

class ValueType:
    """
    Feature value type. Used to define data types in Feature Tables.
    """

    UNKNOWN = ''
    BYTES = 'bytes'
    STRING = 'str'
    INT32 = 'int32'
    INT64 = 'int'
    DOUBLE = 'float'
    FLOAT = 'float32'
    BOOL = 'bool'
    DATETIME = 'datetime'
    BYTES_LIST = ''
    STRING_LIST = ''
    INT32_LIST = ''
    INT64_LIST = ''
    DOUBLE_LIST = ''
    FLOAT_LIST = ''
    BOOL_LIST = ''


def pd_schema_to_value_type(value):
    type_map = {
        "integer": ValueType.INT64,
        "string": ValueType.STRING,
        "number": ValueType.DOUBLE,
        "datetime": ValueType.DATETIME,
    }
    return type_map[value]



def py_to_value_type(value):
    type_name = type(value).__name__
    type_map = {
        "int": ValueType.INT64,
        "str": ValueType.STRING,
        "float": ValueType.DOUBLE,
        "bytes": ValueType.BYTES,
        "float64": ValueType.DOUBLE,
        "float32": ValueType.FLOAT,
        "int64": ValueType.INT64,
        "uint64": ValueType.INT64,
        "int32": ValueType.INT32,
        "uint32": ValueType.INT32,
        "uint8": ValueType.INT32,
        "int8": ValueType.INT32,
        "bool": ValueType.BOOL,
        "timedelta": ValueType.INT64,
        "datetime64[ns]": ValueType.INT64,
        "datetime64[ns, tz]": ValueType.INT64,
        "category": ValueType.STRING,
    }

    if type_name in type_map:
        return type_map[type_name]
