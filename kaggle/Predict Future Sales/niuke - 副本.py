while True:
    try:
        num = int(input('请输入个数：'))
    except:
        print('键值对的个数是整形!')
        continue

    if not isinstance(num, int):
        print('键值对的个数是整形！')
        continue
    if num <=0:
        print('键值对的个数必须大于0！')

    lst = []
    tmp_dict = {}
    for i in range(num):

        while True:
            row = input('请输入键值对')
            rowlst = row.split()

            if len(rowlst) != 2:
                print('只能包含一个空格')
                continue

            try:
                k = int(rowlst[0])
                v = int(rowlst[1])
            except:
                print('k,v是整形!')
                continue

            if not isinstance(k, int) or k < 0:
                print('key值必须是大于等于0的整数')
                continue

            if not isinstance(v, int) or v <= 0:
                print('value值必须是大于0的整数')
                continue

            break

        if k not in tmp_dict:
            tmp_dict[k] = v
        else:
            tmp_dict[k] = tmp_dict[k] + v

    tmp_dict2 = dict(sorted(list(tmp_dict.items()), key=lambda x:x[0], reverse=False))
    print(tmp_dict2)
    break


