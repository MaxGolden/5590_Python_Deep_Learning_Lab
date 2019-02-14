"""
Lab 1 - 1
Write a program that computes the net amount of a bank account based a transaction log from console input. The
transaction log format is shown as following:
   Suppose the following input is supplied to the program:
   Deposit 300
   Deposit 250
   Withdraw 100
   Deposit 50
   Then the output should be
   Total amount - $500

"""


class Account:

    account_amount = 0

    def __init__(self, n):
        self.name = n

    def deposit(self, n, value):
        if self.name == n:
            Account.account_amount = self.account_amount + value
        else:
            print("Name is not found")

    def withdraw(self, n, value):
        if self.name == n:
            Account.account_amount = self.account_amount - value
        else:
            print("Name is not found")


if __name__ == "__main__":

    account1 = Account(max)

    print('Deposit 300')
    account1.deposit(max, 300)

    print('Deposit 250')
    account1.deposit(max, 250)

    print('Withdraw 100')
    account1.withdraw(max, 100)

    print('Deposit 50\n')
    account1.deposit(max, 50)

    print("Total amount -", account1.account_amount)

