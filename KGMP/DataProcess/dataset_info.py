import pickle as pkl
path_data = '../DataProcess/data_manage_key_mapping.pkl'


def info(path=None):
    if path is None:
        path = path_data
    data = pkl.load(open(path, mode='rb'))
    # print(data)
    author = []
    keys = []
    author_dict = dict()
    keys_dict = dict()
    for cur_data in data:
        for j in cur_data:
            a1 = j[0]
            a2 = j[1]
            keys_ = j[2]
            # print(a1, a2, keys_)
            for a_tmp in [a1, a2]:
                if a_tmp not in author:
                    author.append(a_tmp)
            for cur_key in keys_:
                if cur_key not in keys:
                    keys.append(cur_key)
    # print(len(author), author)
    # print(len(keys), keys)
    for x in author:
        if x not in author_dict.keys():
            author_dict[x] = len(author_dict)
    for y in keys:
        if y not in keys_dict.keys():
            keys_dict[y] = len(keys_dict)
    return author_dict, keys_dict, len(author), len(keys)

#
# if __name__ == "__main__":
#     info()
