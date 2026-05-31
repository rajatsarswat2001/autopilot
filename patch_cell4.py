import json

with open("autopilot_kaggle.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

cell4 = next(c for c in notebook['cells']
             if c['cell_type'] == 'code' and c.get('id') == 'cell-4-run')

new_source = []
for line in cell4['source']:
    if 'GPU mode (Wan2.1 ON)' in line:
        line = line.replace('GPU mode (Wan2.1 ON)', 'GPU mode (CogVideoX-2B)')
    if 'Wan2.1 + Chatterbox' in line:
        line = line.replace('Wan2.1 + Chatterbox', 'CogVideoX-2B + Chatterbox')
    if 'WAN21_ENABLED=0' in line:
        line = line.replace('WAN21_ENABLED=0', 'VIDEO_GEN_ENABLED=0')
    if 'WAN21_ENABLED' in line:
        line = line.replace('WAN21_ENABLED', 'VIDEO_GEN_ENABLED')
    new_source.append(line)

cell4['source'] = new_source

with open("autopilot_kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Cell 4 patched.")
