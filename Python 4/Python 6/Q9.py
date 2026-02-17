class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def display(self):
        print(f"Name: {self.name}, Age: {self.age}")

class Student(Person):
    def __init__(self, name, age, student_id, grade):
        # Call base class constructor
        super().__init__(name, age)
        self.student_id = student_id
        self.grade = grade

    def display_student(self):
        # Inherit functionality from Person
        self.display()
        print(f"ID: {self.student_id}, Grade: {self.grade}")

# Create object and demonstrate inheritance
student = Student("John Doe", 20, "S12345", "A")
student.display_student()
