
def sum_integers(x, y):
    sum_val = x + y
    if sum_val in range(15, 20):
        return 20
    return sum_val

a = int(input("Input first integer: "))
b = int(input("Input second integer: "))
print(sum_integers(a, b))
