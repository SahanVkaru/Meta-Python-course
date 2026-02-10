
try:
    temp_input = input("Input the temperature you like to convert? (e.g., 45F, 102C etc.) : ")
    degree = int(temp_input[:-1])
    convention = temp_input[-1]

    if convention.upper() == "C":
        result = int(round((9 * degree) / 5 + 32))
        o_convention = "Fahrenheit"
        print("The temperature in", o_convention, "is", result, "degrees.")
    elif convention.upper() == "F":
        result = int(round((degree - 32) * 5 / 9))
        o_convention = "Celsius"
        print("The temperature in", o_convention, "is", result, "degrees.")
    else:
        print("Input proper convention.")
except ValueError:
    print("Invalid input format.")
