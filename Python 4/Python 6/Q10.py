class BankAccount:
    def __init__(self, initial_balance=0):
        self.__balance = initial_balance  # Private attribute

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            print(f"Deposited: ${amount}")
        else:
            print("Invalid deposit amount")

    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            print(f"Withdrew: ${amount}")
        else:
            print("Insufficient funds or invalid amount")

    def get_balance(self):
        return self.__balance

# Testing
account = BankAccount(1000)
account.deposit(500)
account.withdraw(200)
print(f"Final Balance: ${account.get_balance()}")
