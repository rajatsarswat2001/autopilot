import json

with open("autopilot_kaggle.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if "enable_attention_slicing" in line:
                continue # Remove this line
            new_source.append(line)
        cell['source'] = new_source

with open("autopilot_kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook patched to remove enable_attention_slicing().")
