from mlrun.errors import err_to_str


def test_error_none():
    assert err_to_str(None) == ""


def test_error_is_already_string():
    assert err_to_str("this is already a string") == "this is already a string"


def test_error_single():
    try:
        raise Exception("a")
    except Exception as ex:
        assert err_to_str(ex) == "a"


def test_error_chain_n2():
    try:
        raise Exception("b") from Exception("a")
    except Exception as ex:
        assert err_to_str(ex) == "b, caused by: a"


def test_error_chain_n3():
    try:
        a = Exception("a")
        b = Exception("b")
        b.__cause__ = a
        raise Exception("c") from b
    except Exception as ex:
        assert err_to_str(ex) == "c, caused by: b, caused by: a"


def test_error_circular_chain():
    a = Exception("a")
    b = Exception("b")
    a.__cause__ = b
    b.__cause__ = a
    assert err_to_str(b) == "b, caused by: a"
