
def spam(list1, list2):
    return len(set(list1) & set(list2))

def max_by_length(dct):
    max_key = None
    max_value = None
    max_length = -1

    for key, value in dct.items():
        if len(value) > max_length:
            max_key = key
            max_value = value
            max_length = len(value)

    return (max_key, max_value)

def generate_lst_long(r, c):
    result = []
    for i in range(r):
        row = []
        for j in range(c):
            row.append(i + j)
        result.append(row)
    return result




