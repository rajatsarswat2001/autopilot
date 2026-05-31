import json

with open("autopilot_kaggle.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and 'id' in cell and cell['id'] == 'cell-2-gpu-models':
        new_source = []
        skip_mode = False
        for line in cell['source']:
            if "try:" in line and "torchao.quantization" in cell['source'][cell['source'].index(line)+1]:
                skip_mode = True
            
            if skip_mode:
                if "except Exception as qe:" in line:
                    continue
                if "INT8 skipped" in line:
                    skip_mode = False
                    continue
                continue
                
            new_source.append(line)
        cell['source'] = new_source

    if cell['cell_type'] == 'code' and 'id' in cell and cell['id'] == 'cell-3-keys':
        new_source = []
        for line in cell['source']:
            if "'VIDEO_GEN_INT8':" in line:
                new_source.append("    'VIDEO_GEN_INT8':         '0',   # Disabled to prevent torchao crashes\n")
            elif "'VIDEO_GEN_STRATEGY':" in line and "'mirror'" in line:
                new_source.append("    'VIDEO_GEN_STRATEGY':     'hybrid' if (has_gpu and torch.cuda.device_count() >= 2) else 'sequential',\n")
            else:
                new_source.append(line)
        cell['source'] = new_source

with open("autopilot_kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook patched successfully (removed torchao and changed strategy)!")
