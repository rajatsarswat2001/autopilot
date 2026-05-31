import json

with open("autopilot_kaggle.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and 'id' in cell and cell['id'] == 'cell-2-gpu-models':
        new_source = []
        for line in cell['source']:
            if "pipe.to('cuda:{video_gpu_id}')" in line:
                new_source.extend([
                    "if hasattr(pipe, 'enable_model_cpu_offload'):\n",
                    "    pipe.enable_model_cpu_offload(gpu_id={video_gpu_id})\n",
                    "else:\n",
                    "    pipe.to('cuda:{video_gpu_id}')\n"
                ])
            else:
                new_source.append(line)
        cell['source'] = new_source

with open("autopilot_kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook patched successfully!")
