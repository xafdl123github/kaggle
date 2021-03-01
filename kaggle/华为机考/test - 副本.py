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


def find_prime(even_lst, odd_lst):
    lines = []
    for r in range(len(even_lst)):  # 行
        row_primes = []
        for c in odd_lst:  # 列
            if judge_prime(r + c):
                row_primes.append(c)

        lines.append(len(row_primes))

    # ic(lines)

    if not lines:
        return 0

    if max(lines) == 0:
        return 0

    lines = list(filter(lambda x:x!=0, lines))

    # 在列表中找到最小数的索引
    r_index = lines.index(min(lines))  # 如果存在相同的数，则取第一次出现的索引
    min_row = even_lst[r_index]  # 最小偶数

    del even_lst[r_index]

    for key, c in enumerate(odd_lst):
        if judge_prime(min_row + c):  # 如果是素数
            del odd_lst[key]
            return 1 + find_prime(even_lst, odd_lst)

    return 0


while True:
    # try:
    row1 = input()
    num = int(row1) # 数据个数

    row2 = input()
    num_lst = row2.split()
    num_lst = list(map(int, num_lst))   # 转为整形

    # 偶数和奇数的个数不一定相等
    evens = list(filter(judge_oushu, num_lst))   # 偶数列表   行
    odds = list(filter(judge_qishu, num_lst))    # 奇数列表   列

    prime_pair_num = find_prime(evens, odds)

    # ic(evens, odds)

    # if len(evens) == len(odds):
    #     prime_pair_num = find_prime(evens, odds)
    # elif len(evens) > len(odds):
    #     prime_pair_num = find_prime(evens[:len(odds)], odds)
    #     yushu = evens[len(odds):]
    #     new_evens = yushu[:int(len(yushu)/2)]
    #     new_odds = yushu[int(len(yushu)/2):]
    #     prime_pair_num += find_prime(new_evens, new_odds)
    # else:
    #     prime_pair_num = find_prime(evens, odds[:len(evens)])
    #     ic(prime_pair_num)
    #     yushu = odds[len(evens):]
    #     ic(len(yushu), len(yushu) / 2)
    #     new_evens = yushu[:int(len(yushu) / 2)]
    #     new_odds = yushu[int(len(yushu) / 2):]
    #     prime_pair_num += find_prime(new_evens, new_odds)
    #     ic(prime_pair_num)

    print(prime_pair_num)


    # except:
    #     break