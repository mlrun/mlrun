from copy import copy


class ChildChain:
    def __init__(self, context, name=None, secret=None):
        self.context = context
        self.name = name
        self.secret_key = secret

    def do(self, x):
        x = copy(x)
        secret_value = self.context.get_secret(self.secret_key)

        x.append({f"child:{self.secret_key}": secret_value})
        return x
