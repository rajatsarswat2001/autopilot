import json

with open("autopilot_kaggle.ipynb", "r", encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if source and "CELL 2 -- PRE-LOAD GPU MODELS IN SUBPROCESS" in source[1]:
            # Patch Wan2.1 smoke test string to be an f-string
            new_source = []
            for line in source:
                if 'wan_code = """\n' in line:
                    new_source.append('    wan_code = f"""\n')
                elif "pipe.to(f'cuda:{video_gpu_id}')\n" in line:
                    new_source.append("pipe.to('cuda:{video_gpu_id}')\n")
                elif "pipe.vae.enable_slicing()\n" in line:
                    new_source.append(line)
                    new_source.append("pipe.vae.enable_tiling()\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source

        elif source and "CELL 3 — API KEYS & SETTINGS" in source[1]:
            # Add step counts to settings
            new_source = []
            for line in source:
                if "    'VIDEO_GEN_ENABLED':          '1' if has_gpu else '0',   # disable on CPU-only\n" in line:
                    new_source.append(line)
                    new_source.append("    'VIDEO_GEN_WAN_STEPS':    '50',\n")
                    new_source.append("    'VIDEO_GEN_LTX_STEPS':    '40',\n")
                    new_source.append("    'VIDEO_GEN_COG_STEPS':    '50',\n")
                    new_source.append("    'DISABLE_STOCK':          '1',\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source
            
        elif source and "CELL 4 — RUN PIPELINE" in source[1]:
            # Change free < 10 to free < 8
            new_source = []
            for line in source:
                if "    if free < 10:\n" in line:
                    new_source.append("    if free < 8:\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open("autopilot_kaggle.ipynb", "w", encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully.")
