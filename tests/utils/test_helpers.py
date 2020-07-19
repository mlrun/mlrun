from mlrun.utils.helpers import verify_field_regex
from mlrun.utils.regex import run_name


def test_run_name_regex():
    cases = [
        {'value': 'asd', 'valid': True},
        {'value': 'asd', 'valid': True},
        {'value': 'asa', 'valid': True},
        {'value': 'as-123-2-8a', 'valid': True},
        {'value': '1as-123-2-8a5', 'valid': True},
        {
            'value': 'azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azs',
            'valid': True,
        },
        {
            # Invalid because the first letter is -
            'value': '-as-12328a',
            'valid': False,
        },
        {
            # Invalid because the last letter is .
            'value': 'as-12328a-',
            'valid': False,
        },
        {
            # Invalid because $ is not allowed
            'value': 'as-12328A$a',
            'valid': False,
        },
        {
            # Invalid because it's more then 63 characters
            'value': 'azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsx',
            'valid': False,
        },
    ]
    for case in cases:
        try:
            verify_field_regex('test_field', case['value'], run_name)
        except Exception:
            if case['valid']:
                raise
