def fun(n):
    if n <= 2:
        return 1
    elif n == 3:
        return 2
    else:
        return fun(n-1) + fun(n-2)

while True:
    try:
        n = int(input())
        print(fun(n))
    except:
        break