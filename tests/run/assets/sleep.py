import datetime
import time


def handler(context, time_to_sleep=1):
    context.log_result("started", str(datetime.datetime.now()))
    time.sleep(int(time_to_sleep))
    context.log_result("finished", str(datetime.datetime.now()))
