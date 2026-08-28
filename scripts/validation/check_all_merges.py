import openpyxl, os

base = r'c:\Users\madeinweb\Documents\GitHub\projeto-nidec-cad-review\nidec-cad-review\scripts\validation\41-50 Structured reviews'
files = [f for f in os.listdir(base) if f.endswith('.xlsx')]
files.sort()

for fname in files:
    wb = openpyxl.load_workbook(os.path.join(base, fname))
    ws = wb['Single Drawing Data Extraction']
    merges = list(ws.merged_cells.ranges)
    print(f'{fname}: {merges}')
