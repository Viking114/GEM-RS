def getdata(lang='zh'):
    path = r'C:\Users\talki\PycharmProjects\fuckyou\test_data\conll2012.train.txt' if lang == 'zh' else \
        r'C:\Users\talki\PycharmProjects\fuckyou\test_data\en.txt'
    text = open(path, encoding='utf-8').read()
    text = text.split('\n')
    result = []
    for t in text:
        result.append(t.split('|||')[0].split(' ')[1:])

    return result

def gettext8():
    text = open(r'C:\Users\talki\PycharmProjects\fuckyou\test_data\text8', encoding='utf-8').read()
    text = text.split('\n')
    text = text[0].split(' ')
    result = []
    s,e = 0, 1000
    while e < len(text):
        result.append(text[s:e])
        s += 1000
        e += 1000

    return result