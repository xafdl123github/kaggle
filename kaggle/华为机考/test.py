from icecream import ic

m = 7  # 苹果数
n = 3  # 盘子数

def putApple(m, n):
    if m == 0 or n == 1:
        return 1

    if m < n:   # 苹果数 < 盘子数
        # 没有空盘子
        return putApple(m, m)

    else:  # 苹果数7 >= 盘子数3
        # ++++++++++++++++++++有空盘子
        # putApple(m, n-1)：有一个盘子是空的
        # putApple(m-n, n)：每一个盘子最小有一个苹果
        return putApple(m, n-1) + putApple(m-n, n)

res = putApple(m, n)
