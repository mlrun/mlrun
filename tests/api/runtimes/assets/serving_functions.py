from copy import copy


class Chain:
    def __init__(self, context, name=None, secret=None):
        self.context = context
        self.name = name
        self.secret_name = secret

    def do(self, x):
        x = copy(x)
        secret_value = self.context.get_secret(self.secret_name)

        x.append({self.secret_name: secret_value})
        return x
