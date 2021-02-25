import random
import re

# num = random.randint(2, 5)
num = 3

lst = []
for i in range(num):
    row = input()
    lst.append(row)

for i in lst:
    find_empty = re.fullmatch('\s+', i)
    if find_empty:  # 如果是空字符串，不处理
        print(i)
    else:  # 如果不是空字符串
        if len(i) < 8:
            eight = '{}{}'.format(i, '0' * (8 - len(i)))
            print(eight)
        elif len(i) == 8:
            print(i)
        else:  # 大于8位
            substr_num = len(i) // 8  # 8的倍数
            left_num = len(i) % 8  # 8的余数
            # 循环倍数
            for k in range(0, substr_num):
                print(i[k * 8: (k + 1) * 8])

            if left_num != 0:
                eight = '{}{}'.format(i[substr_num*8], '0' * (8 - left_num + 1))
                print(eight)