def is_base64_code(s):
    '''Check s is Base64.b64encode'''
    if not isinstance(s, str) or not s:
        return "params s not string or None"

    _base64_code = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a',
                    'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                    't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1',
                    '2', '3', '4', '5', '6', '7', '8', '9', '+',
                    '/', '=']
    _base64_code_set = set(_base64_code)  # 转为set增加in判断时候的效率
    # Check base64 OR codeCheck % 4
    code_fail = [i for i in s if i not in _base64_code_set]
    if code_fail or len(s) % 4 != 0:
        return False
    return True


if __name__ == '__main__':
    print(is_base64_code("c2RhZGFzZGFkYWQ="))
