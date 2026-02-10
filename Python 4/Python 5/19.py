
def check_range(n, start, end):
    if start <= n <= end:
        print(f"{n} is in the range [{start}, {end}]")
    else:
        print(f"{n} is not in the range [{start}, {end}]")

num = int(input("Enter a number: "))
# Example range 1-100
check_range(num, 1, 100)
