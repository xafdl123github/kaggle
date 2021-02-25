def judge(number):
    if number % 2 != 0:
        for i in range(3, number, 2):
            if number % i == 0:
                return True, i
    else:
        for i in range(2, number, 2):
            if number % i == 0:
                return True, i
    return False, number


prime = []

digit = int(input())
while True:
    res = judge(digit)
    if res[0]:
        prime.append(res[1])
        digit //= res[1]
        continue
    else:
        prime.append(res[1])
        break

prime = sorted(prime)
prime = [str(i) for i in prime]

kz = ' '
if len(prime) > 1:
    res = ' '.join(prime)
    print('{}{}'.format(res, kz))
else:
    print('{}{}'.format(str(prime[0]), kz))
