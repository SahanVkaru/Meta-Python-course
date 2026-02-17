# Inheritance
class Person:
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f"Hello, my name is {self.name}."
    
class Student(Person):
    def study(self):
        print (self.name, "is studying.")

stu = Student ("Nimal")
print (stu.greet())
stu.study()