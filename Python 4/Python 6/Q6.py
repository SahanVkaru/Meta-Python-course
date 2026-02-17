while True:
    try:
        user_input = input("Please enter an integer: ")
        result = int(user_input)
        print(f"Success! You entered the integer: {result}")
        break  # Stop only when valid integer is entered
    except ValueError:
        print("Invalid value! Please enter a whole number (integer).")
