import json

with open("autopilot_kaggle.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and 'id' in cell and cell['id'] == 'cell-2-gpu-models':
        new_source = []
        for line in cell['source']:
            if "num_frames=49," in line:
                line = line.replace("num_frames=49,", "num_frames=25,")
            new_source.append(line)
        cell['source'] = new_source

with open("autopilot_kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook patched successfully (reduced frames)!")
