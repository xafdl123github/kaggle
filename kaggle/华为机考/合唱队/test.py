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

        res_lst = []
        for idx in range(1, N-1):   # 循环每一个元素

            # right  186 186 150 200 160 130 197 200  ++++++++++++++++++++++++
            right_con = []
            right_seq = stu_lst[idx + 1:]
            right_xiao_cur = []
            for right_idx in range(len(right_seq)):     # 循环当前元素右侧每一个元素
                if right_seq[right_idx] < stu_lst[idx]:   # 如果 右侧元素 < 当前元素
                    right_xiao_cur.append(right_seq[right_idx])

            right_final_lst = []
            for k in range(len(right_xiao_cur)):
                dandiao_dijian_right = [right_xiao_cur[k]]
                for i in range(k+1, len(right_xiao_cur)):
                    if right_xiao_cur[i] < dandiao_dijian_right[-1]:
                        dandiao_dijian_right.append(right_xiao_cur[i])
                right_final_lst.append(len(dandiao_dijian_right))

            if right_final_lst:
                right_max = max(right_final_lst)   # 右侧
            else:
                right_max = 0

            # left  186 186 150 200 160 130 197 200  ++++++++++++++++++++++++++
            left_seq = stu_lst[:idx]
            left_seq = left_seq[::-1]
            left_xiao_cur = []
            for left_idx in range(len(left_seq)):  # 循环当前元素左侧的每一个元素
                if left_seq[left_idx] < stu_lst[idx]:  # 如果 左侧元素 < 当前元素
                    left_xiao_cur.append(left_seq[left_idx])
            left_final_lst = []
            for k in range(len(left_xiao_cur)):
                dandiao_dijian_left = [left_xiao_cur[k]]
                for i in range(k+1, len(left_xiao_cur)):
                    if left_xiao_cur[i] < dandiao_dijian_left[-1]:
                        dandiao_dijian_left.append(left_xiao_cur[i])
                left_final_lst.append(len(dandiao_dijian_left))

            if left_final_lst:
                left_max = max(left_final_lst)  # 左侧
            else:
                left_max = 0

            res_lst.append(right_max + left_max + 1)

        print(N - max(res_lst))

    except Exception as e:
        break