import json

with open("autopilot_kaggle.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Print Cell 4 full content
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and cell.get('id') == 'cell-4-run':
        src = "".join(cell.get('source', []))
        print(src.encode('ascii', errors='replace').decode('ascii'))
