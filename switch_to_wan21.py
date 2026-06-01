import json
import os

# 1. Update .env
env_path = r'C:\Users\rajat\Downloads\yt\autopilot_pipeline\.env'
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        lines = f.readlines()
    with open(env_path, 'w') as f:
        for line in lines:
            if line.startswith('VIDEO_GEN_MODEL='):
                f.write('VIDEO_GEN_MODEL=wan21\n')
            else:
                f.write(line)

# 2. Update autopilot_kaggle.ipynb
nb_path = r'C:\Users\rajat\Downloads\yt\autopilot_kaggle.ipynb'
if os.path.exists(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell.get('id') == 'cell-3-keys':
            source = cell['source']
            for i, line in enumerate(source):
                if line.startswith("    'VIDEO_GEN_MODEL'"):
                    source[i] = "    'VIDEO_GEN_MODEL':        'wan21',\n"
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
