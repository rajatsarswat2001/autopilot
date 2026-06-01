import json

def patch_notebook():
    with open("autopilot_kaggle.ipynb", "r", encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            if source and "CELL 2 -- PRE-LOAD GPU MODELS IN SUBPROCESS" in source[1]:
                # Patch Wan2.1 smoke test string to include NF4 text encoder
                new_source = []
                inside_wan_code = False
                for line in source:
                    if 'wan_code = f"""\n' in line or 'wan_code = """\n' in line:
                        inside_wan_code = True
                        new_source.append('    wan_code = f"""\n')
                        new_source.append("import os, torch, gc\n")
                        new_source.append("os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n")
                        new_source.append("from diffusers import WanPipeline\n")
                        new_source.append("from transformers import UMT5EncoderModel, BitsAndBytesConfig\n")
                        new_source.append("print('      Loading Wan2.1 T2V-1.3B (NF4 Text Encoder)...')\n")
                        new_source.append("quant_config = BitsAndBytesConfig(\n")
                        new_source.append("    load_in_4bit=True,\n")
                        new_source.append("    bnb_4bit_compute_dtype=torch.float16,\n")
                        new_source.append("    bnb_4bit_quant_type='nf4',\n")
                        new_source.append(")\n")
                        new_source.append("text_encoder = UMT5EncoderModel.from_pretrained(\n")
                        new_source.append("    'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',\n")
                        new_source.append("    subfolder='text_encoder',\n")
                        new_source.append("    quantization_config=quant_config,\n")
                        new_source.append("    device_map={{'': 'cuda:{video_gpu_id}'}},\n")
                        new_source.append("    torch_dtype=torch.float16,\n")
                        new_source.append(")\n")
                        new_source.append("pipe = WanPipeline.from_pretrained(\n")
                        new_source.append("    'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',\n")
                        new_source.append("    text_encoder=text_encoder,\n")
                        new_source.append("    torch_dtype=torch.float16,\n")
                        new_source.append(")\n")
                        new_source.append("pipe.to('cuda:{video_gpu_id}')\n")
                        new_source.append("pipe.enable_attention_slicing()\n")
                        new_source.append("pipe.vae.enable_slicing()\n")
                        new_source.append("pipe.vae.enable_tiling()\n")
                        new_source.append("print('      Running 5-step smoke test...')\n")
                        new_source.append("with torch.no_grad():\n")
                        new_source.append("    pipe(prompt='a calm ocean wave, cinematic',\n")
                        new_source.append("         width=832, height=480, num_frames=9,\n")
                        new_source.append("         num_inference_steps=5, guidance_scale=5.0)\n")
                        new_source.append("print('      Smoke test completed successfully!')\n")
                        new_source.append("del pipe; del text_encoder; gc.collect(); torch.cuda.empty_cache()\n")
                        new_source.append('"""\n')
                        continue
                    if inside_wan_code:
                        if '"""\n' in line:
                            inside_wan_code = False
                        continue
                    new_source.append(line)
                cell['source'] = new_source

    with open("autopilot_kaggle.ipynb", "w", encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook patched.")

if __name__ == "__main__":
    patch_notebook()
