import xlrd
import re
import pickle as pkl
data_path = '../Source/all.xlsx'


def read_xlsx(path=None):
    if path is None:
        path = data_path
    workbook = xlrd.open_workbook(path)
    booksheet = workbook.sheet_by_name('Sheet1')
    p = list()
    for row in range(booksheet.nrows):
        if row == 0:
            continue
        else:
            row_data = []
            for col in range(booksheet.ncols):
                cel = booksheet.cell(row, col)
                val = cel.value
                try:
                    val = cel.value
                    val = re.sub(r'\s+', '', val)
                except Exception as ex:
                    pass
                if type(val) == float:
                    val = int(val)
                else:
                    val = str(val)
                row_data.append(val)
            print(row_data)
            p.append(row_data)
    pkl.dump(p, open('all.pkl', mode='wb'))

    return p


def get_author_article_mapping(pkl_path='all.pkl'):
    data = pkl.load(open(pkl_path, mode="rb"))
    data_save = []
    for cur_data in data:
        author, article = cur_data[3:]
        author = str(author).split(',')
        article = str(article).split(',')
        author_pair = []
        if len(author) == 1:
            author_pair = [[author[0], 'A_NA', article]]
        else:
            for j in range(1, len(author), 1):
                author_pair.append([author[j - 1], author[j], article])
        print(author_pair)
        print(article)
        data_save.append(author_pair)
        print(' '.join(author) + '\t' + ' '.join(article))
        print('*' * 100)
    pkl.dump(data_save, open('data_manage.pkl', mode='wb'))


def article_mapping():
    data = pkl.load(open('data_manage.pkl', mode='rb'))
    max_key = 0
    for j in data:
        for k in j:
            if len(k[-1]) > max_key:
                max_key = len(k[-1])
    print('max_key:', max_key)
    # mapping
    for index_data in range(len(data)):
        for index_index_data in range(len(data[index_data])):
            print(data[index_data][index_index_data][-1])
            if len(data[index_data][index_index_data][-1]) < (max_key * .6):
                data[index_data][index_index_data][-1] = data[index_data][index_index_data][-1] + \
                                                    (int(max_key * .6) - len(data[index_data][index_index_data][-1])) * ["PAD"]
            print(data[index_data][index_index_data][-1])
            print('*' * 100)
    pkl.dump(data, open('data_manage_key_mapping.pkl', mode='wb'))


def get_more():
    # older_data = pkl.load(open('data_manage_key_mapping.pkl', mode='rb'))
    www = pkl.load(open('all.pkl', mode='rb'))
    data = dict()
    unique_author = []
    unique_year = []
    for cur_w_index, cur_w in enumerate(www):
        _, _, year, author, keywords = cur_w[:]
        if year not in unique_year:
            unique_year.append(year)
        if isinstance(author, int):
            author_ = [[str(author), str(author)]]
        else:
            author_ = []
            tmp_a = author.strip().split(',')
            for j_index in range(0, len(tmp_a) - 1, 1):
                author_.append([tmp_a[0], tmp_a[j_index + 1]])
        keywords = keywords.strip().split(',')
        if (author_[0][0], year) not in data.keys():
            data[(author_[0][0], year)] = [[author_, keywords]]
        else:
            data[(author_[0][0], year)].append([author_, keywords])
    # get all useful data
    all_q = []
    print(data.keys)
    print(data[('3', 2011)])
    for cur_ke in data.keys():
        if cur_ke[0] not in unique_author:
            unique_author.append(cur_ke[0])
    # clip the data dataset into year sequence
    max_length_pairs = 0
    # all_data_cur = []
    # useful_qqq = []
    for cur_au in unique_author:
        qqq = []
        useful_qqq = []
        for cur_year in data.keys():  # year
            if cur_au in cur_year:
                qqq.append([cur_year, data[cur_year]])
        all_data_cur = sorted(qqq, key=lambda k: k[0][-1])
        length_cur = 0
        for cur_all_data_cur in all_data_cur:
            useful_qqq.append(cur_all_data_cur[-1])
            if length_cur < len(cur_all_data_cur):
                length_cur = len(cur_all_data_cur)

        for qq in range(len(useful_qqq)):
            useful_qqq[qq] = useful_qqq[qq] + useful_qqq[qq] * (length_cur - len(useful_qqq[qq]))
        print('12345')
        wwww = zip(*useful_qqq)
        for jj in wwww:
            print(jj)
        print('#' * 100)
    
    # print('all_data_cur sorted:', all_data_cur)
    # print('useful_qqq sorted:', useful_qqq)
    # for cur_a in unique_author:


get_more()

qq = [1, 2]
qqq = [6, 8]
qqqq = [16, 18, 26]




print("yes")
"""
[[('177', 2011), [[['177', '178']], ['294', '295', '296']]], 
[('177', 2014), [[['177', '2254']], ['2033', '1754', '3171', '3172'], [[['177', '10358']], ['5741', '1754', '14708', '3343', '3763']], 
                [[['177', '14988']], ['5742', '23743', '23744', '12421']], [[['177', '15028']], ['668', '1144', '9720', '3171']], [[['177', '15076']], ['5', '1754', '2288']]]], 
[('177', 2015), [[['177', '2743']], ['2033', '1754', '3831'], [[['177', '6071']], ['19347', '3589', '19348', '19349']], [[['177', '15325']], ['6166', '24528', '4459']], 
                [[['177', '15343']], ['24551', '3588', '24552', '24553']]]], 
[('177', 2016), [[['177', '3197']], ['2392', '4459', '4460', '4461'], 
                [[['177', '10897']], ['15727', '3957', '1172', '8719', '11326', '15728']], [[['177', '12808']], ['14608', '9424', '19500', '19501']], 
                [[['177', '7247']], ['1237', '8941', '25215', '8437']]]], 
[('177', 2019), [[['177', '3546']], ['342', '6166', '6167', '6168'], 
                [[['177', '7224']], ['7806', '8284', '8818', '7222']], [[['177', '7257']], ['6166', '1755', '8501', '8863']], 
                [[['177', '7322']], ['1755', '8941', '8942', '6990']]]], 
[('177', 2017), [[['177', '177']], ['2033', '8264', '1755', '262'], 
                [[['177', '12869']], ['9837', '19668', '15007']]]], 
[('177', 2018), [[['177', '7023']], ['8563', '8284', '4459', '6912']]], 
[('177', 2020), [[['177', '1623']], ['9291', '6166', '6168', '2330', '9292', '9293']]], 
[('177', 2012), [[['177', '9650']], ['678', '149', '302', '266', '418'], 
                [[['177', '14130']], ['153', '152', '21881']]]], 
[('177', 2013), [[['177', '177']], ['1754', '13593', '2033', '19', '9915'], 
                [[['177', '11349']], ['345', '3831', '5', '6495']]]]]
# #####################################################################################################################
[[('177', 2011), [[['177', '178']], ['294', '295', '296']]], 
[('177', 2012), [[['177', '9650']], ['678', '149', '302', '266', '418'], [[['177', '14130']], ['153', '152', '21881']]]], 
[('177', 2013), [[['177', '177']], ['1754', '13593', '2033', '19', '9915'], [[['177', '11349']], ['345', '3831', '5', '6495']]]], 
[('177', 2014), [[['177', '2254']], ['2033', '1754', '3171', '3172'], [[['177', '10358']], ['5741', '1754', '14708', '3343', '3763']], [[['177', '14988']], ['5742', '23743', '23744', '12421']], [[['177', '15028']], ['668', '1144', '9720', '3171']], [[['177', '15076']], ['5', '1754', '2288']]]], 
[('177', 2015), [[['177', '2743']], ['2033', '1754', '3831'], [[['177', '6071']], ['19347', '3589', '19348', '19349']], [[['177', '15325']], ['6166', '24528', '4459']], [[['177', '15343']], ['24551', '3588', '24552', '24553']]]], 
[('177', 2016), [[['177', '3197']], ['2392', '4459', '4460', '4461'], [[['177', '10897']], ['15727', '3957', '1172', '8719', '11326', '15728']], [[['177', '12808']], ['14608', '9424', '19500', '19501']], [[['177', '7247']], ['1237', '8941', '25215', '8437']]]], 
[('177', 2017), [[['177', '177']], ['2033', '8264', '1755', '262'], [[['177', '12869']], ['9837', '19668', '15007']]]], 
[('177', 2018), [[['177', '7023']], ['8563', '8284', '4459', '6912']]], 
[('177', 2019), [[['177', '3546']], ['342', '6166', '6167', '6168'], [[['177', '7224']], ['7806', '8284', '8818', '7222']], [[['177', '7257']], ['6166', '1755', '8501', '8863']], [[['177', '7322']], ['1755', '8941', '8942', '6990']]]], 
[('177', 2020), [[['177', '1623']], ['9291', '6166', '6168', '2330', '9292', '9293']]]]


[[('177', 2011), [[[['177', '178']], ['294', '295', '296']]]], 
[('177', 2014), [[[['177', '2254']], ['2033', '1754', '3171', '3172']], [[['177', '10358']], ['5741', '1754', '14708', '3343', '3763']], [[['177', '14988']], ['5742', '23743', '23744', '12421']], [[['177', '15028']], ['668', '1144', '9720', '3171']], [[['177', '15076']], ['5', '1754', '2288']]]], 
[('177', 2015), [[[['177', '2743']], ['2033', '1754', '3831']], [[['177', '6071']], ['19347', '3589', '19348', '19349']], [[['177', '15325']], ['6166', '24528', '4459']], [[['177', '15343']], ['24551', '3588', '24552', '24553']]]], 
[('177', 2016), [[[['177', '3197']], ['2392', '4459', '4460', '4461']], [[['177', '10897']], ['15727', '3957', '1172', '8719', '11326', '15728']], [[['177', '12808']], ['14608', '9424', '19500', '19501']], [[['177', '7247']], ['1237', '8941', '25215', '8437']]]], 
[('177', 2019), [[[['177', '3546']], ['342', '6166', '6167', '6168']], [[['177', '7224']], ['7806', '8284', '8818', '7222']], [[['177', '7257']], ['6166', '1755', '8501', '8863']], [[['177', '7322']], ['1755', '8941', '8942', '6990']]]], 
[('177', 2017), [[[['177', '177']], ['2033', '8264', '1755', '262']], [[['177', '12869']], ['9837', '19668', '15007']]]], 
[('177', 2018), [[[['177', '7023']], ['8563', '8284', '4459', '6912']]]], 
[('177', 2020), [[[['177', '1623']], ['9291', '6166', '6168', '2330', '9292', '9293']]]], 
[('177', 2012), [[[['177', '9650']], ['678', '149', '302', '266', '418']], [[['177', '14130']], ['153', '152', '21881']]]], 
[('177', 2013), [[[['177', '177']], ['1754', '13593', '2033', '19', '9915']], [[['177', '11349']], ['345', '3831', '5', '6495']]]]]


"""
