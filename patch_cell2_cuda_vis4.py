import json

def patch():
    with open("autopilot_kaggle.ipynb", "r", encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and any("CELL 2 -- PRE-LOAD GPU MODELS IN SUBPROCESS" in line for line in cell['source']):
            source = cell['source']
            for i, line in enumerate(source):
                if line == "    env1 = {**os.environ,\n":
                    # It's currently:
                    # "    env1 = {**os.environ,\n"
                    # "            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',\n"
                    # "            'TOKENIZERS_PARALLELISM': 'false',\n"
                    # "            'TRANSFORMERS_VERBOSITY': 'error'}\n"
                    if "TOKENIZERS_PARALLELISM" in source[i+2]:
                        source.insert(i+2, "            'CUDA_VISIBLE_DEVICES': str(video_gpu_id),\n")
                
                if line == '    wan_code = f"""\n':
                    # Need to replace lines starting from wan_code down to the end of the string
                    wan_start = i
                    wan_end = i
                    for j in range(i+1, len(source)):
                        if source[j] == '"""\n':
                            wan_end = j
                            break
                    
                    new_wan = [
                        '    wan_code = f"""\n',
                        "import os, torch, gc\n",
                        "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n",
                        "from diffusers import WanPipeline\n",
                        "pipe = WanPipeline.from_pretrained(\n",
                        "    'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',\n",
                        "    torch_dtype=torch.float16,\n",
                        ")\n",
                        "pipe.to('cuda:0')\n",
                        "pipe.enable_attention_slicing()\n",
                        "pipe.vae.enable_slicing()\n",
                        "pipe.vae.enable_tiling()\n",
                        "with torch.no_grad():\n",
                        "    pipe(prompt='a calm ocean wave, cinematic',\n",
                        "         width=832, height=480, num_frames=9,\n",
                        "         num_inference_steps=5, guidance_scale=5.0)\n",
                        "print('Smoke test passed')\n",
                        "del pipe; gc.collect(); torch.cuda.empty_cache()\n",
                        '"""\n'
                    ]
                    source[wan_start:wan_end+1] = new_wan
                    break
            
            cell['source'] = source
            break

    with open("autopilot_kaggle.ipynb", "w", encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("Patched via JSON")

if __name__ == "__main__":
    patch()
