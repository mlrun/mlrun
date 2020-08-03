import pathlib

from tests.system.base import TestMLRunSystem


class TestMlRunExamples(TestMLRunSystem):
    artifacts_path = pathlib.Path(__file__).absolute().parent / 'artifacts'
