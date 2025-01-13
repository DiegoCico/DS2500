def clean_empty(lst):
    new_lst = []
    for i in lst:
        try:
            new_lst.append(int(i))
        except ValueError:
            new_lst.append(0)
    return new_lst

def sum_avg(lst):
    sum = 0
    for i in lst:
        sum += i
    average = sum / len(lst)
    return (sum, average)

def lst_to_dct(lst):
    dct = {}
    for i in lst:
        if len(i) == 1:
            dct[i[0]] = []
        elif len(i) == 2:
            dct[i[0]] = [i[1]]
        elif len(i) == 3:
            dct[i[0]] = [i[1], i[2]]
    return dct

print(lst_to_dct([[1, 2], [4, 5]]))