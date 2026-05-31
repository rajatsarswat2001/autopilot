"""
Patch Cell 3 of autopilot_kaggle.ipynb to:
- Set PYTORCH_CUDA_ALLOC_CONF (correct name) instead of PYTORCH_ALLOC_CONF (wrong name that does nothing)
- Remove VIDEO_GEN_INT8 (dead code — _USE_INT8 no longer exists in video_gen_tools.py)
"""
import json, re

with open("autopilot_kaggle.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

cell3 = None
for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and cell.get('id') == 'cell-3-pipeline-config':
        cell3 = cell
        break

if cell3 is None:
    # Try to find it by content
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            src = "".join(cell.get('source', []))
            if 'VIDEO_GEN_STRATEGY' in src and 'NICHE' in src:
                cell3 = cell
                break

if cell3 is None:
    print("ERROR: Cell 3 not found!")
else:
    new_source = []
    for line in cell3['source']:
        # Fix wrong env var name (silently did nothing)
        if 'PYTORCH_ALLOC_CONF' in line and 'CUDA' not in line:
            line = line.replace('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF')
            print(f"Fixed env var name: {line.rstrip()}")
        # Remove dead VIDEO_GEN_INT8 lines
        if 'VIDEO_GEN_INT8' in line:
            print(f"Removed dead code: {line.rstrip()}")
            continue
        new_source.append(line)

    # If PYTORCH_CUDA_ALLOC_CONF not set anywhere in Cell 3, inject it
    src_joined = "".join(new_source)
    if 'PYTORCH_CUDA_ALLOC_CONF' not in src_joined:
        # Find where os.environ assignments begin and inject before them
        inject_line = 'os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # correct name (not PYTORCH_ALLOC_CONF)\n'
        for i, line in enumerate(new_source):
            if 'os.environ' in line and 'VIDEO_GEN' in line:
                new_source.insert(i, inject_line)
                print(f"Injected PYTORCH_CUDA_ALLOC_CONF at line {i}")
                break

    cell3['source'] = new_source
    print("Cell 3 patched.")

with open("autopilot_kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("Notebook saved.")
