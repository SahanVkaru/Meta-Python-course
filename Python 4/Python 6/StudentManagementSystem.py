import os

class Student:
    def __init__(self, sid, name, grade):
        self.sid = sid
        self.name = name
        self.grade = grade

    def __str__(self):
        return f"ID: {self.sid} | Name: {self.name} | Grade: {self.grade}"

def add_student():
    try:
        sid = input("Enter Student ID: ")
        name = input("Enter Name: ")
        grade = input("Enter Grade: ")
        
        with open("students.txt", "a") as f:
            f.write(f"{sid},{name},{grade}\n")
        print("Student added successfully!")
    except Exception as e:
        print(f"Error saving student: {e}")

def view_students():
    if not os.path.exists("students.txt"):
        print("No student records found.")
        return

    print("\n--- Student Records ---")
    try:
        with open("students.txt", "r") as f:
            for line in f:
                sid, name, grade = line.strip().split(",")
                s = Student(sid, name, grade)
                print(s)
    except Exception as e:
        print(f"Error reading records: {e}")
    print("------------------------")

def main():
    while True:
        print("\n--- Student Management System ---")
        print("1. Add Student")
        print("2. View Students")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ")
        
        if choice == '1':
            add_student()
        elif choice == '2':
            view_students()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
