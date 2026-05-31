import json

with open("autopilot_kaggle.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and 'id' in cell and cell['id'] == 'cell-2-gpu-models':
        new_source = []
        for line in cell['source']:
            if "cog_code = f\"\"\"" in line:
                new_source.append(line)
                new_source.append("import os\n")
                new_source.append("os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'\n")
                continue
            
            # Remove eager for CogVideoX because we WANT it to use memory efficient SDPA!
            # We keep it for Chatterbox because Chatterbox needs it.
            if "os.environ['TRANSFORMERS_ATTN_IMPLEMENTATION'] = 'eager'" in line:
                # Let's just check if it's in the CogVideoX block
                if "cog_code = " in "".join(new_source):
                    # We are inside cog_code string. Remove it.
                    continue
            
            new_source.append(line)
        cell['source'] = new_source

with open("autopilot_kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook patched successfully!")
