try:
    num1 = input("Enter the first number: ")
    num2 = input("Enter the second number: ")
    
    # Convert inputs to floats to handle both integers and decimals
    val1 = float(num1)
    val2 = float(num2)
    
    result = val1 / val2
    print(f"Result: {val1} / {val2} = {result}")

except ValueError:
    print("Error: Invalid input. Please enter numeric values.")
except ZeroDivisionError:
    print("Error: Division by zero is not allowed.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
