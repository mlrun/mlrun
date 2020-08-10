import pathlib

from tests.system.base import TestMLRunSystem


class TestMLRunExamples(TestMLRunSystem):
    assets_path = pathlib.Path(__file__).absolute().parent / 'assets'
