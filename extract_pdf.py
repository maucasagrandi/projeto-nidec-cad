import fitz

pdf_path = r'c:\Users\madeinweb\Documents\GitHub\projeto-nidec-cad-review\nidec-cad-review\CADS\13358002_REV_7_draw_1.pdf'
out_path = r'c:\Users\madeinweb\Documents\GitHub\projeto-nidec-cad-review\nidec-cad-review\CADS\13358002_REV_7_draw_1.txt'

doc = fitz.open(pdf_path)
total_pages = len(doc)
print(f'Total pages: {total_pages}')

all_text = []
for i, page in enumerate(doc):
    text = page.get_text()
    all_text.append(f'=== PAGE {i+1} ===\n{text}')
    print(f'Page {i+1}: {len(text)} chars')

full_text = '\n'.join(all_text)
print(f'Total chars: {len(full_text)}')

with open(out_path, 'w', encoding='utf-8') as f:
    f.write(full_text)

print('Saved!')
doc.close()
