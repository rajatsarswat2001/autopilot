import json

with open("autopilot_kaggle.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and cell.get('id') == 'cell-3-keys':
        src = "".join(cell.get('source', []))
        print(src.encode('ascii', errors='replace').decode('ascii'))
