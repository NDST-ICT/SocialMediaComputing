import time


if __name__ == '__main__':
    journals = ['PR', 'PRA', 'PRB', 'PRC', 'PRD',
                'PRE', 'PRI', 'PRL', 'PRSTAB',
                'PRSTPER', 'RMP']

    journals_cas = ['PR', 'PRL', 'PRB']

    ti_table = dict()
    for jour in journals:
        file = open('time_list_' + jour + '.txt', 'r')
        line = file.readline()
        while line:
            name, dt = line.split(',')
            dt = dt[:-1]
            if dt == '' or dt[5:7] == '00':
                line = file.readline()
                continue
            if dt[-2:] == '00':
                dt = dt[0:-2] + '01'
            dt = str(int(dt[0:4]) + 400) + dt[4:]
            timeArray = time.strptime(dt, "%Y-%m-%d")
            timestamp = time.mktime(timeArray)
            timestamp = timestamp / 3600
            timestamp = (timestamp - 16) / 24
            timestamp = int(timestamp)
            ti_table[name] = timestamp
            line = file.readline()
        file.close()

    cas = dict()
    file = open('./aps/citing_cited.csv', 'r')
    line = file.readline()
    while line:
        citing, cited = line.split(',')
        cited = cited[:-1]
        if (citing not in ti_table) or (cited not in ti_table):
            line = file.readline()
            continue
        citing_ti = ti_table[citing]
        cited_ti = ti_table[cited]
        if cited not in cas:
            cas[cited] = []
        at_ti = citing_ti - cited_ti
        if at_ti > 0:
            cas[cited].append(at_ti)
        line = file.readline()
    file.close()

    for jour in journals_cas:
        file = open('time_list_' + jour + '.txt', 'r')
        out = open('cas_' + jour + '.txt', 'w')
        line = file.readline()
        while line:
            name, dt = line.split(',')
            year = int(dt[0:4])
            if (name not in cas) or (jour == 'PR' and year not in range(1960, 1970)) or \
                    (jour == 'PRL' and year not in range(1970, 1980)) or \
                    (jour == 'PRB' and year not in range(1980, 1990)):
                line = file.readline()
                continue
            cas[name].sort()
            one_cas = cas[name]
            at_in_five_years = 0
            for at_ti in one_cas:
                if at_ti <= 365 * 5:
                    at_in_five_years = at_in_five_years + 1
                else:
                    break
            if at_in_five_years <= 10:
                line = file.readline()
                continue
            one_cas_len = len(one_cas)
            for i in range(one_cas_len-1):
                out.write(str(one_cas[i]) + '\t')
            out.write(str(one_cas[one_cas_len - 1]) + '\n')
            line = file.readline()
        file.close()
        out.close()
