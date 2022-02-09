def hello_world(context):
    context.logger.info("hello world")


class mycls:
    def __init__(self, context=None, a1=1):
        self.context = context
        self.a1 = a1

    def mtd(self, context, x=0, y=0):
        print(f"x={x}, y={y}, a1={self.a1}")
        context.log_result("rx", x)
        context.log_result("ry", y)
        context.log_result("ra1", self.a1)
