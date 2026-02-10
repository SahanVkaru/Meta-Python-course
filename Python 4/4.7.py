def count_down(start):
    next_number = start-1
    if next_number>0:
        count_down(next_number)

count_down(5)