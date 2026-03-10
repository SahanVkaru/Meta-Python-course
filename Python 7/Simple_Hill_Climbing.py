def f(x):
    return -x**2 + 4*x

def hill_climbing(function, start, step=0.1, max_iter=100):
    current = start
    for _ in range(max_iter):
        neighbors = [current + step, current - step]
        next_state = max(neighbors, key=function)
        if function(next_state) <= function(current):
            return current
        current = next_state
    return current

result = hill_climbing(f, start=0)
print("Hill Climbing Result:", result)
print("Function Value:", f(result))