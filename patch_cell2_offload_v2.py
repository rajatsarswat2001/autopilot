import json

def patch():
    with open("autopilot_kaggle.ipynb", "r", encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            for i, line in enumerate(source):
                if line == "pipe.to('cuda:0')\\n":
                    source[i] = "pipe.enable_model_cpu_offload()\\n"
            cell['source'] = source

    with open("autopilot_kaggle.ipynb", "w", encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("Notebook patched to use CPU offload")

if __name__ == "__main__":
    patch()
