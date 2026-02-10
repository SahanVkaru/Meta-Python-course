
def is_palindrome(s):
    return s == s[::-1]

s = input("Enter a string: ")
if is_palindrome(s):
    print("The string is a palindrome")
else:
    print("The string is not a palindrome")
