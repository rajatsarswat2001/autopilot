import json

with open("autopilot_kaggle.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and 'id' in cell and cell['id'] == 'cell-2-gpu-models':
        new_source = []
        skip = False
        for line in cell['source']:
            if "from torchao.quantization" in line:
                # We hit the torchao block. Remove the previous "try:\n" that we already appended.
                if new_source and new_source[-1].strip() == "try:":
                    new_source.pop()
                skip = True
                continue
            
            if skip:
                if "print('      INT8 skipped" in line:
                    skip = False
                continue
                
            new_source.append(line)
        cell['source'] = new_source

with open("autopilot_kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook patched successfully!")
