import json

with open("autopilot_kaggle.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if "pipe.enable_vae_slicing()" in line:
                # Replace with safe version or just remove it if it's the smoke test
                line = line.replace("pipe.enable_vae_slicing()", "if hasattr(pipe, 'enable_vae_slicing'): pipe.enable_vae_slicing()")
            if "pipe.enable_vae_tiling()" in line:
                line = line.replace("pipe.enable_vae_tiling()", "if hasattr(pipe, 'enable_vae_tiling'): pipe.enable_vae_tiling()")
            new_source.append(line)
        cell['source'] = new_source

with open("autopilot_kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook patched to use hasattr() for vae slicing.")
