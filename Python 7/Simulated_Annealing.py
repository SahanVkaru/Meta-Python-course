import math
import random

def f(x):
    return -x**2 + 4*x

def simulated_annealing(function, start, temp=100, cooling=0.95, max_iter=1000):
    current = start
    best = current
    for i in range(max_iter):
        neighbor = current + random.uniform(-1,1)
        delta = function(neighbor) - function(current)
        if delta > 0 or random.random() < math.exp(delta/temp):
            current = neighbor
        if function(current) > function(best):
            best = current
        temp *= cooling
    return best

result = simulated_annealing(f, start=random.uniform(-5,5))
print("Simulated Annealing Result:", result)
print("Function Value:", f(result))