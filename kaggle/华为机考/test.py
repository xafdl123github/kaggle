import re

while True:
    try:
        seq = input()

        res = re.findall(r'[a-zA-Z]+', seq)
        if not res:
            break

        lst = res[::-1]
        print(' '.join(lst))
    except:
        break