# 检查ip是否合法
def check_ip(ip):  # ip=['192','168','0','1']
    if len(ip) != 4 or '' in ip:
        return False
    else:
        for c in ip:
            if int(c) < 0 or int(c) > 255:
                return False
        return True

# 检查ms是否合法(数值大小必须满足前面是连续的1，后面是连续的0，全1或者全0都非法)
lll = ['254','252','248','240','224','192','128','0']

def check_ms(ms):  # ms=['255','255','255','0']
    if len(ms) != 4 or '' in ms:
        return False
    if ms[0] == '255':
        if ms[1] == '255':
            if ms[2] == '255':
                if ms[3] in lll:
                    return True
                else:
                    return False
            elif ms[2] in lll and ms[3] == '0':
                return True
            else:
                return False
        elif ms[1] in lll and ms[2] == '0' and ms[3] == '0':
            return True
        else:
            return False
    elif ms[0] in lll[:-1] and ms[1] == '0' and ms[2] == '0' and ms[3] == '0':
        return True
    else:
        return False

def trans_to_bin(ip_or_ms):  # ip_or_ms=['192','168','0','1']
    bin_str = ''
    for field in ip_or_ms:
        bin_str += format(int(field), 'b').zfill(8)
    return bin_str

def ip_and_ms(ip_bin, ms_bin):
    and_str = ''  # ip二进制串与ms二进制串相与后的二进制串
    for i in range(32):
        if ip_bin[i] == '1' and ms_bin[i] == '1':
            and_str += '1'
        else:
            and_str += '0'
    return and_str

# 主函数部分
while True:
    try:
        ms = input().strip().split('.')  # ms=['255','255','255','0']
        ip1 = input().strip().split('.')  # ip1=['192','168','0','1']
        ip2 = input().strip().split('.')

        # 若ip1,ip2和ms都合法，才进一步判断ip1与ip2是否在同一个子网
        if check_ms(ms) and check_ip(ip1) and check_ip(ip2):
            ms_bin = trans_to_bin(ms)
            ip1_bin = trans_to_bin(ip1)
            ip2_bin = trans_to_bin(ip2)

            str1 = ip_and_ms(ip1_bin, ms_bin)
            str2 = ip_and_ms(ip2_bin, ms_bin)

            if str1 == str2:
                print(0)
            else:
                print(2)
        else:
            print(1)
    except:
        break
