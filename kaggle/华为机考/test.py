from icecream import ic


# 判断偶数
def judge_oushu(digit):
    if digit % 2 == 0:
        return True
    else:
        return False

# 判断奇数
def judge_qishu(digit):
    if digit % 2 != 0:
        return True
    else:
        return False

# 判断素数
def judge_prime(digit):
    if digit % 2 == 0:  # 偶数
        return False
    else:   # 奇数
        for i in range(2, digit//2):
            if digit % i == 0:
                return False
        return True

while True:
    try:
        row1 = input()
        num = int(row1) # 数据个数

        row2 = input()
        num_lst = row2.split()
        num_lst = list(map(int, num_lst))   # 转为整形

        # 偶数和奇数的个数不一定相等
        oushu_lst = filter(judge_oushu, num_lst)   # 偶数列表   行
        qishu_lst = filter(judge_qishu, num_lst)    # 奇数列表   列

        lines = []
        for r in range(oushu_lst):   # 行
            row_primes = []
            for c in qishu_lst:    # 列
                if judge_prime(r + c):
                    row_primes.append(c)

            lines.append(len(row_primes))

        # 在列表中找到最小数的索引
        r_index = lines.index(min(lines))   # 如果存在相同的数，则取第一次出现的索引






    except:
        break