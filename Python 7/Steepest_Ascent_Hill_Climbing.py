def f(x):
    return -x**2 + 4*x

def steepest_ascent(function, start, step=0.1, max_iter=100):
    current = start
    for _ in range(max_iter):
        neighbors = [current + i*step for i in range(-5,6)]
        next_state = max(neighbors, key=function)
        if function(next_state) <= function(current):
            break
        current = next_state
    return current

result = steepest_ascent(f, start=0)
print("Steepest Ascent Result:", result)
print("Function Value:", f(result))