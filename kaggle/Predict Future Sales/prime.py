digit = int(input())

def get_prime(num):

    i = 2

    while i*i <= num:
        while num % i == 0:
            print(i, end=' ')
            num = num // i
        i += 1

    if num - 1 != 0:
        print(num, end=' ')

    return num

get_prime(digit)