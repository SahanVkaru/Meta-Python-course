
with open("Student.txt", "w") as f:
    f.write("Name: John Doe\n")
    f.write("Name: Jane Smith\n")
    f.write("Name: Alice Johnson\n")

with open("Student.txt", "a") as f:
    f.write("Name: Duvidu putha\n")
    f.write("Name: Laki putha\n")
    f.write("Name: Nepul putha\n")


with open("Student.txt") as f:
    lines = f.read().splitlines()
    print(lines)


