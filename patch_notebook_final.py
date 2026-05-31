import json

with open("autopilot_kaggle.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Replace model_cpu_offload with sequential_cpu_offload
            if "pipe.enable_model_cpu_offload" in line:
                line = line.replace("enable_model_cpu_offload", "enable_sequential_cpu_offload")
            
            # Add attention_slicing right after cpu_offload
            new_source.append(line)
            if "enable_sequential_cpu_offload" in line:
                new_source.append("    pipe.enable_attention_slicing(slice_size=1)\n")
            
            # Ensure VAE slicing/tiling is called on pipe.vae
            if "if hasattr(pipe, 'enable_vae_slicing'): pipe.enable_vae_slicing()" in line:
                new_source.remove(line)
            elif "pipe.enable_vae_slicing()" in line:
                new_source.remove(line)
            elif "if hasattr(pipe, 'enable_vae_tiling'): pipe.enable_vae_tiling()" in line:
                new_source.remove(line)
            elif "pipe.enable_vae_tiling()" in line:
                new_source.remove(line)
        
        # Add the VAE lines safely
        if any("CogVideoXPipeline" in l for l in new_source):
            idx = next((i for i, l in enumerate(new_source) if "print('      Running 5-step smoke test...')" in l), -1)
            if idx != -1:
                new_source.insert(idx, "    pipe.vae.enable_slicing()\n")
                new_source.insert(idx, "    pipe.vae.enable_tiling()\n")
        
        cell['source'] = new_source

with open("autopilot_kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook patched for sequential offload, attention slicing, and VAE.")
