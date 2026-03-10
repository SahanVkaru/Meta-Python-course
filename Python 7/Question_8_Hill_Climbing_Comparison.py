import math
import random

# Function to optimize
def f(x):
    return -x**2 + 4*x

def simple_hill_climbing(function, start, step=0.1, max_iter=100):
    current = start
    iterations = 0
    
    for _ in range(max_iter):
        iterations += 1
        neighbors = [current + step, current - step]
        next_state = max(neighbors, key=function)
        
        if function(next_state) <= function(current):
            break
        current = next_state
    
    return {
        "solution": current,
        "function_value": function(current),
        "iterations": iterations
    }

def steepest_ascent_hill_climbing(function, start, step=0.1, max_iter=100):
    current = start
    iterations = 0
    
    for _ in range(max_iter):
        iterations += 1
        neighbors = [current + i*step for i in range(-5, 6)]
        next_state = max(neighbors, key=function)
        
        if function(next_state) <= function(current):
            break
        current = next_state
    
    return {
        "solution": current,
        "function_value": function(current),
        "iterations": iterations
    }

def simulated_annealing(function, start, temp=100, cooling=0.95, max_iter=1000):
    current = start
    best = current
    iterations = 0
    
    for _ in range(max_iter):
        iterations += 1
        neighbor = current + random.uniform(-1, 1)
        delta = function(neighbor) - function(current)
        
        if delta > 0 or random.random() < math.exp(delta/temp):
            current = neighbor
        
        if function(current) > function(best):
            best = current
        
        temp *= cooling
    
    return {
        "solution": best,
        "function_value": function(best),
        "iterations": iterations
    }

# Run all three algorithms
start_point = 0

shc_result = simple_hill_climbing(f, start_point)
print("Simple Hill Climbing:")
print(f"  Final solution: {shc_result['solution']:.4f}")
print(f"  Function value: {shc_result['function_value']:.4f}")
print(f"  Iterations used: {shc_result['iterations']}")

sahc_result = steepest_ascent_hill_climbing(f, start_point)
print("\nSteepest Ascent Hill Climbing:")
print(f"  Final solution: {sahc_result['solution']:.4f}")
print(f"  Function value: {sahc_result['function_value']:.4f}")
print(f"  Iterations used: {sahc_result['iterations']}")

sa_result = simulated_annealing(f, start_point)
print("\nSimulated Annealing:")
print(f"  Final solution: {sa_result['solution']:.4f}")
print(f"  Function value: {sa_result['function_value']:.4f}")
print(f"  Iterations used: {sa_result['iterations']}")

print("\nPerformance Comparison:")
print("- Simple Hill Climbing may get stuck in local optima")
print("- Steepest Ascent explores more neighbors but still prone to local optima")
print("- Simulated Annealing can escape local optima and often finds global optimum")