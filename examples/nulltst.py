import sys
import time


def test_job():
    print("im running")
    print("args:", sys.argv)

    time.sleep(0.1)
    print("finish")


if __name__ == "__main__":
    test_job()
