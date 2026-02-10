
import random

target_num = random.randint(1, 9)
guess_num = 0

print("Guess a number between 1 and 9")

while target_num != guess_num:
    try:
        guess_num = int(input('Enter your guess: '))
        if guess_num == target_num:
            print('Well guessed!')
        else:
            print("Try again!")
    except ValueError:
        print("Please enter a valid number.")
