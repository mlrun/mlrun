from mlrun.errors import error_to_string


def test_error_none():
    assert error_to_string(None) == ""


def test_error_is_already_string():
    assert error_to_string("this is already a string") == "this is already a string"


def test_error_single():
    try:
        raise Exception("a")
    except Exception as ex:
        assert error_to_string(ex) == "a"


def test_error_chain_n2():
    try:
        raise Exception("b") from Exception("a")
    except Exception as ex:
        assert error_to_string(ex) == "b, caused by: a"


def test_error_chain_n3():
    try:
        a = Exception("a")
        b = Exception("b")
        b.__cause__ = a
        raise Exception("c") from b
    except Exception as ex:
        assert error_to_string(ex) == "c, caused by: b, caused by: a"


def test_error_circular_chain():
    a = Exception("a")
    b = Exception("b")
    a.__cause__ = b
    b.__cause__ = a
    assert error_to_string(b) == "b, caused by: a"
