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
        # 186 186 150 187 200 160 130 197 200  188  190 199 200
        # 1    1   1       2   2   1   3   4
        for i in range(len(stu_lst)):
            cur_stu = stu_lst[i]   # 当前元素
            ascend = [cur_stu]  # 递增容器
            for j in range(len(stu_lst)):
                if i != j:
                    if stu_lst[j] > ascend[-1]:   # 如果右侧元素比当前元素大
                        ascend.append(stu_lst[j])


        
        # 每个数（从右侧开始）在最大递增子序列中的位置，从1开始

    except:
        break