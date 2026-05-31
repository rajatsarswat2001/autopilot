import json

with open("autopilot_kaggle.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Print all cell IDs and first 200 chars of source
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        src = "".join(cell.get('source', []))[:200]
        print(f"=== Cell {i} id={cell.get('id','NO-ID')} ===")
        print(src.encode('ascii', errors='replace').decode('ascii'))
        print()
