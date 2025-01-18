def dct_max(dct):
    '''
    :param dct: one dict
    :return: tuple with two items
    does: finds the max valie and its dictionary
        returns the value and its key
    '''
    max = float("-inf")
    max_key = 0
    for key, val in dct.items():
        if val > max:
            max = val
            max_key = key

    return (max_key, max)

def sum_list(dct):
    sum = 0
    for key, val in dct.items():
        sum += val
    return sum

def main():
    ct = dct_max({"a": 1, "b": 2, "c": 3})
    sum = sum_list({"a": 1, "b": 2, "c": 3})
    print(ct)
    print(sum)

if __name__ == "__main__":
    main()