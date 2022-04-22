import mlrun


def myhandler(context: mlrun.MLClientCtx, x=4):
    print(f"Run: {context.name} (uid={context.uid})")
    context.log_result("y", x * 2)
