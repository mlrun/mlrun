
import time
def handler(context, seconds: int = 5):
    context.logger.info(f"Sleeping {seconds}s …")
    time.sleep(seconds)
