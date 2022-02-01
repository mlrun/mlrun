import mlrun


def myhandler(context: mlrun.MLClientCtx, p1, p2=3, p3=4):
    print(f"Run: {context.name} (uid={context.uid})")
    context.logger.info(f"iter={context.iteration} p2={p2}")
    context.log_result("accuracy", p2 * 2)
    context.log_result("loss", p3 * 3)
    context.log_artifact("file_result", body=b"abc123", local_path="result.txt")


class mycls:
    def __init__(self, context=None, a1=1):
        self.context = context
        self.a1 = a1

    def mtd(self, context, x=0, y=0):
        print(f"x={x}, y={y}, a1={self.a1}")
        context.log_result("rx", x)
        context.log_result("ry", y)
        context.log_result("ra1", self.a1)
