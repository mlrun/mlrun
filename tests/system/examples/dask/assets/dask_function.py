# function that will be distributed
def inc(x):
    return x + 2


# wrapper function, uses the dask client object
def main(context, x=1, y=2):
    context.logger.info("params: x={},y={}".format(x, y))
    print("params: x={},y={}".format(x, y))
    x = context.dask_client.submit(inc, x)
    print(x)
    print(x.result())
    context.log_result("y", x.result())
