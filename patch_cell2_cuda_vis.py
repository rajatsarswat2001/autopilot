import json

def patch_cell2():
    with open("autopilot_kaggle.ipynb", "r", encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and "CELL 2 -- PRE-LOAD GPU MODELS IN SUBPROCESS" in "".join(cell['source']):
            new_source = []
            skip_wan = False
            for line in cell['source']:
                if "    env1 = {**os.environ," in line:
                    new_source.append(line)
                    new_source.append("            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',\n")
                    new_source.append("            'CUDA_VISIBLE_DEVICES': str(video_gpu_id),\n")
                    new_source.append("            'TOKENIZERS_PARALLELISM': 'false',\n")
                    new_source.append("            'TRANSFORMERS_VERBOSITY': 'error'}\n")
                    skip_wan = True # Skip the next few lines until wan_code
                    continue
                
                if skip_wan:
                    if "    wan_code = " in line:
                        skip_wan = False
                    else:
                        continue
                
                if "    wan_code = " in line:
                    new_source.append('    wan_code = f"""\n')
                    new_source.append("import os, torch, gc\n")
                    new_source.append("os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n")
                    new_source.append("from diffusers import WanPipeline\n")
                    new_source.append("pipe = WanPipeline.from_pretrained(\n")
                    new_source.append("    'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',\n")
                    new_source.append("    torch_dtype=torch.float16,\n")
                    new_source.append(")\n")
                    new_source.append("pipe.to('cuda:0')\n")
                    new_source.append("pipe.enable_attention_slicing()\n")
                    new_source.append("pipe.vae.enable_slicing()\n")
                    new_source.append("pipe.vae.enable_tiling()\n")
                    new_source.append("with torch.no_grad():\n")
                    new_source.append("    pipe(prompt='a calm ocean wave, cinematic',\n")
                    new_source.append("         width=832, height=480, num_frames=9,\n")
                    new_source.append("         num_inference_steps=5, guidance_scale=5.0)\n")
                    new_source.append("print('Smoke test passed')\n")
                    new_source.append("del pipe; gc.collect(); torch.cuda.empty_cache()\n")
                    new_source.append('"""\n')
                    skip_wan = True
                    continue
                
                if skip_wan and '"""\n' in line:
                    skip_wan = False
                    continue
                
                if not skip_wan:
                    new_source.append(line)
                    
            cell['source'] = new_source
            break

    with open("autopilot_kaggle.ipynb", "w", encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    patch_cell2()
