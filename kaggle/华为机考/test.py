def issu(x):
    tem = 2
    while tem ** 2 <= x:
        if x % tem == 0:
            return False
        tem += 1
    return True


def find(ji_number, l1, l2, ou):
    for i in range(0, len(ou)):
        if issu(ji_number + ou[i]) and l1[i] == 0:
            l1[i] = 1
            if l2[i] == 0 or find(l2[i], l1, l2, ou):
                l2[i] = ji_number
                return True
    return False


try:
    while True:
        n = input()
        n = int(n)
        l = list(map(int, input().split()))
        ji, ou = [], []
        for i in range(n):
            if l[i] % 2 == 0:
                ou.append(l[i])
            else:
                ji.append(l[i])
        result = 0
        match = [0] * len(ou)
        for i in range(0, len(ji)):
            used = [0] * len(ou)
            if find(ji[i], used, match, ou):
                result += 1
        print(result)
except:
    pass