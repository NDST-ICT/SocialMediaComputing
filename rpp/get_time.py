import xml.etree.ElementTree as ET
import time


if __name__ == '__main__':
    journals = ['PR', 'PRA', 'PRB', 'PRC', 'PRD',
                'PRE', 'PRI', 'PRL', 'PRSTAB',
                'PRSTPER', 'RMP']

    for jour in journals:
        out = open('time_list_' + jour + '.txt', 'w')
        tree = ET.parse('./aps/' + jour + '.xml')
        aticles = tree.getroot()
        for art in aticles:
            dt = art.find('issue').attrib['printdate']
            out.write(art.attrib['doi'] + ',')
            out.write(dt + '\n')
        print('complete ' + jour)
        out.close()
