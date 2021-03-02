from icecream import ic
import re
import sys

while 1:
    try:
        N = int(input())  # 同学数量

        sec_row = input()
        stu_lst = sec_row.split()  # 同学列表

        if N != len(stu_lst):
            continue

        # 每个数（从左侧开始）在最大递增子序列中的位置，从1开始

        
        # 每个数（从右侧开始）在最大递增子序列中的位置，从1开始

    except:
        break