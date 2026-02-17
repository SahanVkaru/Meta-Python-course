# Mini integrated task - Student Record Management System
class Student:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks

    def calculate_total(self):
        return sum(self.marks)

    def calculate_average(self):
        return sum(self.marks) / len(self.marks)

# Data entry
name = input("Enter student name: ")
marks = [int(input(f"Enter marks for subject {i+1}: ")) for i in range(3)]

# Create student object and save to file
student = Student(name, marks)
with open("record.txt", "a") as file:
    file.write(f"{name}, {marks[0]}, {marks[1]}, {marks[2]}\n")

# Read file and display summary
print("\n===== STUDENT RECORDS SUMMARY =====")
try:
    with open("record.txt", "r") as file:
        for line in file:
            parts = line.strip().split(", ")
            if len(parts) == 4:
                student_name = parts[0]
                student_marks = [int(parts[1]), int(parts[2]), int(parts[3])]
                student_obj = Student(student_name, student_marks)
                print(f"\nName: {student_obj.name}")
                print(f"Marks: {student_obj.marks}")
                print(f"Total: {student_obj.calculate_total()}")
                print(f"Average: {student_obj.calculate_average():.2f}")
except FileNotFoundError:
    print("No records found yet.")