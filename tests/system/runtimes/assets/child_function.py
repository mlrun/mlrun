class Identity:
    def do(self, x):
        return x


class Augment:
    def do(self, x):
        x["more_stuff"] = 5
        return x
