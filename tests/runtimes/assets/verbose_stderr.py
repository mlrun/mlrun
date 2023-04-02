import sys

print("some output")

for i in range(10000):
    print("123456789", file=sys.stderr)

sys.exit(1)
