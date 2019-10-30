from mlrun import utils


def test_func_info_ann():
    def inc(n: int) -> int:
        """increment n"""
        return n + 1

    out = utils.func_info(inc)
    expected = {
        'name': inc.__name__,
        'doc': inc.__doc__,
        'return': 'int',
        'params': [
            {'name': 'n', 'type': 'int'},
        ],
    }
    assert out == expected, 'inc'


def test_func_info_no_ann():
    def inc(n):
        return n + 1

    out = utils.func_info(inc)
    expected = {
        'name': inc.__name__,
        'doc': '',
        'return': '',
        'params': [
            {'name': 'n', 'type': ''},
        ],
    }
    assert out == expected, 'inc'
