class BaseClass:
    def __init__(self, a: int):
        self.a = a


class InheritingClass(BaseClass):
    def __init__(self, a: int, b: str):
        super().__init__(a=a)
        self.b = b
