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
            # left   186 186 150 200 160 130 197 200
            left_con = []
            for left_idx in range(idx):   # 循环当前元素左侧每一个元素

                if len(left_con) == 0:
                    if stu_lst[left_idx] < stu_lst[idx]:
                        left_con.append(stu_lst[left_idx])
                else:
                    if stu_lst[left_idx] <= left_con[-1]:
                        left_con = []
                        left_con.append(stu_lst[left_idx])
                    else:  # 当前元素比容器（递增序列）最后一个元素大
                        if stu_lst[left_idx] < stu_lst[idx]:
                            left_con.append(stu_lst[left_idx])
                        else:
                            left_con = []

            # right  186 186 150 200 160 130 197 200
            right_con = []
            right_seq = stu_lst[idx + 1:]
            right_seq = right_seq[::-1]   # 当前元素的右边序列（不包含当前元素）
            # ic(right_seq)
            right_xiao_cur = []
            for right_idx in range(len(right_seq)):     # 循环当前元素右侧每一个元素

                if len(right_con) == 0:
                    if right_seq[right_idx] < stu_lst[idx]:
                        # ic(idx, right_seq[right_idx])
                        right_con.append(right_seq[right_idx])
                else:
                    if right_seq[right_idx] <= right_con[-1]:
                        right_con = []
                        right_con.append(right_seq[right_idx])
                    else:  # 当前元素比容器（递增序列）最后一个元素大
                        if right_seq[right_idx] < stu_lst[idx]:
                            right_con.append(right_seq[right_idx])
                        else:
                            right_con = []

            if len(left_con) > 0 and len(right_con) > 0:
                res_lst.append(len(left_con) + len(right_con) + 1)

        print(N - max(res_lst))

    except:
        break