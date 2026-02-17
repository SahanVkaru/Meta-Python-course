class Book:
    def __init__(self, title, author, price):
        self.title = title
        self.author = author
        self.price = price

    def display_details(self):
        print(f"Title: {self.title}")
        print(f"Author: {self.author}")
        print(f"Price: ${self.price}")
        print("-" * 20)

# Create 2 objects
book1 = Book("Python Crash Course", "Eric Matthes", 25.00)
book2 = Book("Automate the Boring Stuff", "Al Sweigart", 30.00)

# Display details
book1.display_details()
book2.display_details()
