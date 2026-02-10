
import string

def is_pangram(str1, alphabet=string.ascii_lowercase):
    alphaset = set(alphabet)
    return alphaset <= set(str1.lower())

s = input("Enter a string: ")
if is_pangram(s):
    print("The string is a pangram")
else:
    print("The string is not a pangram")
