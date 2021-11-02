# function that will be distributed
def inc(x):
    return x + 2


# wrapper function, uses the dask client object
def main(context, x=1, y=2):
    context.logger.info(f"params: x={x},y={y}")
    print(f"params: x={x},y={y}")
    x = context.dask_client.submit(inc, x)
    print(x)
    print(x.result())
    context.log_result("y", x.result())
