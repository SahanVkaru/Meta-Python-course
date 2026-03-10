import random
import math

# Function to optimize
def f(x):
    return -x**2 + 4*x

def simple_hill_climbing(initial_x, step_size, max_iterations):
    current_x = initial_x
    iterations = 0
    
    for _ in range(max_iterations):
        iterations += 1
        current_value = f(current_x)
        
        # Explore neighbors
        neighbor_plus = current_x + step_size
        neighbor_minus = current_x - step_size
        
        value_plus = f(neighbor_plus)
        value_minus = f(neighbor_minus)
        
        if value_plus > current_value:
            current_x = neighbor_plus
        elif value_minus > current_value:
            current_x = neighbor_minus
        else:
            # Local maximum reached
            break
            
    return {
        "final_solution": current_x,
        "function_value": f(current_x),
        "iterations": iterations
    }

def steepest_ascent_hill_climbing(initial_x, step_size, max_iterations):
    current_x = initial_x
    iterations = 0
    
    for _ in range(max_iterations):
        iterations += 1
        current_value = f(current_x)
        
        # Find the best neighbor
        best_neighbor = current_x
        best_value = current_value
        
        for move in [-step_size, step_size]:
            neighbor_x = current_x + move
            neighbor_value = f(neighbor_x)
            if neighbor_value > best_value:
                best_value = neighbor_value
                best_neighbor = neighbor_x
                
        if best_neighbor == current_x:
            # Local maximum reached
            break
        else:
            current_x = best_neighbor
            
    return {
        "final_solution": current_x,
        "function_value": f(current_x),
        "iterations": iterations
    }

def simulated_annealing(initial_x, initial_temp, cooling_rate, max_iterations):
    current_x = initial_x
    current_value = f(current_x)
    temp = initial_temp
    iterations = 0
    
    for _ in range(max_iterations):
        iterations += 1
        
        # Generate a random neighbor
        neighbor_x = current_x + random.uniform(-1, 1)
        neighbor_value = f(neighbor_x)
        
        # Decide whether to move
        if neighbor_value > current_value:
            current_x = neighbor_x
            current_value = neighbor_value
        else:
            acceptance_prob = math.exp((neighbor_value - current_value) / temp)
            if random.random() < acceptance_prob:
                current_x = neighbor_x
                current_value = neighbor_value
                
        # Cool down
        temp *= cooling_rate
        
    return {
        "final_solution": current_x,
        "function_value": f(current_x),
        "iterations": iterations
    }

# Parameters
initial_x = 0
step_size = 0.1
max_iterations = 100
initial_temp = 100
cooling_rate = 0.95

# Run algorithms
shc_results = simple_hill_climbing(initial_x, step_size, max_iterations)
sahc_results = steepest_ascent_hill_climbing(initial_x, step_size, max_iterations)
sa_results = simulated_annealing(initial_x, initial_temp, cooling_rate, max_iterations)

# Print results
print("Simple Hill Climbing:")
print(f"  Final solution: {shc_results['final_solution']:.4f}")
print(f"  Function value: {shc_results['function_value']:.4f}")
print(f"  Iterations: {shc_results['iterations']}")

print("\nSteepest Ascent Hill Climbing:")
print(f"  Final solution: {sahc_results['final_solution']:.4f}")
print(f"  Function value: {sahc_results['function_value']:.4f}")
print(f"  Iterations: {sahc_results['iterations']}")

print("\nSimulated Annealing:")
print(f"  Final solution: {sa_results['final_solution']:.4f}")
print(f"  Function value: {sa_results['function_value']:.4f}")
print(f"  Iterations: {sa_results['iterations']}")

print("\nComparison:")
print("Simple and Steepest Ascent Hill Climbing are prone to getting stuck in local optima.")
print("Simulated Annealing can escape local optima due to its probabilistic nature, often finding a better solution.")
