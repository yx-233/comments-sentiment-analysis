import re
import pandas as pd
from tqdm import tqdm

fileters = ['"', '#', '$', '%', '&', '\(', '\)',  ',', '-', '\.', '/', ':', ';', '<', '>', "\'"
    , '@', '\[', '\]', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~',
            '\t', '\n', '\x97', '\x96', '”', '“','\ue956','\\u200d' ]

def Csv2Txt(input_path,encoding = 'gbk'):
    data = pd.read_csv(input_path, header=None, encoding=encoding)
    output_path = input_path[:-3] + 'txt'
    with open(output_path, 'a+', encoding=encoding) as f:
        for line in tqdm(data.values):
            line = re.sub("|".join(fileters), "", str(line))
            f.write(line + '\n')
    print('success')

def Excel2Txt(input_path,):
    data = pd.read_excel(input_path, header=None)
    output_path = input_path[:-4] + 'txt'
    with open(output_path, 'a+', encoding='utf-8') as f:
        for line in tqdm(data.values):
            line = re.sub("|".join(fileters), "", str(line))
            f.write(line + '\n')
    print('success')

if __name__ == '__main__':
    # Csv2Txt(r"C:\Users\18079\Desktop\课程论文\苏州寒山寺 - 副本.csv",encoding="gbk")
    Excel2Txt(r'C:\Users\18079\Desktop\课程论文\苏州上方山森林动物世界.xlsx')
