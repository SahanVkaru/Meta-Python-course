# Object Oriented Programming (OOP) in Python

#Creating a class and object
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def display_info(self):
        print(f"Name:", self.name)
        print(f"Age:", self.age)


# Create object
s1 = Student ("Nimal", 21)
s1.display_info()
