import openpyxl

files = [
    r'c:\Users\madeinweb\Documents\GitHub\projeto-nidec-cad-review\nidec-cad-review\scripts\validation\41-50 Structured reviews\Drawing Data Extraction - Number_ 41.xlsx',
    r'c:\Users\madeinweb\Documents\GitHub\projeto-nidec-cad-review\nidec-cad-review\scripts\validation\41-50 Structured reviews\Drawing Data Extraction - Number_ 43_.xlsx',
]

for fpath in files:
    wb = openpyxl.load_workbook(fpath)
    ws = wb['Single Drawing Data Extraction']
    print(f'\n{fpath.split(chr(92))[-1]}')
    print('Merged cells:', ws.merged_cells)
    for merge in ws.merged_cells.ranges:
        print(f'  {merge}')
