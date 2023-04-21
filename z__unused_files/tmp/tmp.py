import time
def fibonacci(n):
    if n == 0 or n == 1:
        return n
    else:
        return fibonacci(n - 2) + fibonacci(n - 1)

start = time.time()
for _ in range(100):
    num = fibonacci(25)
stop = time.time()

print(stop - start)