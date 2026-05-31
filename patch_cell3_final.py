import json

with open("autopilot_kaggle.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

cell3 = next(c for c in notebook['cells']
             if c['cell_type'] == 'code' and c.get('id') == 'cell-3-keys')

new_source = []
pytorch_injected = False

for line in cell3['source']:
    # Remove dead VIDEO_GEN_INT8 line
    if 'VIDEO_GEN_INT8' in line:
        print(f"Removed: {line.rstrip()}")
        continue

    # Fix old wrong env var name if present
    if 'PYTORCH_ALLOC_CONF' in line and 'CUDA' not in line:
        line = line.replace('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF')
        print(f"Fixed name: {line.rstrip()}")
        pytorch_injected = True

    # Inject PYTORCH_CUDA_ALLOC_CONF just before the .env write loop
    if not pytorch_injected and "env_path = os.path.join(PIPELINE_DIR, '.env')" in line:
        new_source.append(
            "# Critical: set PYTORCH_CUDA_ALLOC_CONF *here* so Cell 4's subprocess.Popen\n"
        )
        new_source.append(
            "# inherits it via os.environ.copy(). The wrong name 'PYTORCH_ALLOC_CONF' silently does nothing.\n"
        )
        new_source.append(
            "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n"
        )
        print("Injected PYTORCH_CUDA_ALLOC_CONF before .env write")
        pytorch_injected = True

    new_source.append(line)

cell3['source'] = new_source

with open("autopilot_kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Cell 3 patched and notebook saved.")
