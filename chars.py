import json


def strQ2B(ustr: str) -> str:
    '''
    Double-byte to single-byte characters.
    '''
    rstr = []
    for uchar in ustr:
        code = ord(uchar)
        if code == 12288:
            code = 32
        elif code >= 65281 and code <= 65374:
            code -= 65248
        rstr.append(chr(code))
    return ''.join(rstr)


def strB2Q(ustr: str) -> str:
    '''
    Single-byte to double-byte characters.
    '''
    rstr = ''
    for uchar in ustr:
        code = ord(uchar)
        if code == 32:
            code = 12288
        elif code >= 32 and code <= 126:
            code += 65248

        rstr += chr(code)
    return rstr


with open('chn3000.txt', 'r') as f:
    chars = f.read()

chars = set(chars.split())
print('chn3500:', len(chars))

ASCII = ['\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*',
         '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6',
         '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B',
         'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
         'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
         '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f',
         'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
         's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']

# Those that don't have corresponding single-bytes
PUNCT = '。《》“”‘’【】「」、￥…—'

chars.update(set(ASCII))
chars.update(set(PUNCT))
chars = list(chars)

print('extended:', len(chars))

with open('valchr.json', 'w') as f:
    json.dump(chars, f)
