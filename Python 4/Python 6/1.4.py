# Encapsulation
class BankAccount:
    def __init__(self):
        self.__balance = 0  # Private attribute

    def deposit(self, amount):
        self.__balance += amount
    
    def get_balance(self):
        return self.__balance
    
account = BankAccount()
account.deposit(5000)
print("Balance:", account.get_balance())
       